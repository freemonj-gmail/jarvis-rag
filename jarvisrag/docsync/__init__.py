"""Sync volatile remote documents into a local corpus with change detection.

Some authoritative sources (e.g. federal-benefits policy PDFs) change whenever
policy or an administration changes, so committing static copies would ship
stale content. Instead, the *recipe* (document names + URLs) is committed in
``document_sources.json`` while the downloaded files and per-file fetch state
stay local/gitignored under each set's ``corpus_dir``.

``refresh_document_set`` downloads each source and rewrites the corpus file ONLY
when its content actually changed (sha256 of the fetched bytes vs the file on
disk; plus an optional ETag/Last-Modified conditional GET to skip unchanged
downloads entirely). Because an unchanged file keeps its mtime, the extraction
cache stays valid and the index manifest + delta-sync re-embed only the chunks
whose text changed — so Milvus is updated incrementally on real content changes,
not on every refresh.

Leaf module: standard library plus a lazily imported ``httpx``.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

# Browser-like headers: opm.gov rejects a bare user agent, and ssa.gov (Akamai)
# returns 403 + an HTML stub without a full browser header set.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15"
    ),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
}
_STATE_FILENAME = ".docsync_state.json"
_HTTP_TIMEOUT = 60.0
_DEFAULT_CONFIG = "document_sources.json"


@dataclass
class DocSource:
    """One remote document to sync (target filename + source URL)."""
    name: str
    url: str


@dataclass
class DocSet:
    """A named group of remote documents synced into one corpus directory."""
    name: str
    corpus_dir: str
    documents: list[DocSource] = field(default_factory=list)


def load_document_sources(config_path: Optional[str] = None) -> list[DocSet]:
    """Load the committed source recipe. Never raises on missing/invalid file."""
    path = config_path or os.environ.get("JARVIS_DOCUMENT_SOURCES", _DEFAULT_CONFIG)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return []
    sets: list[DocSet] = []
    for entry in raw.get("sets", []) if isinstance(raw, dict) else []:
        try:
            docs = [
                DocSource(name=str(d["name"]), url=str(d["url"]))
                for d in entry.get("documents", [])
                if isinstance(d, dict) and d.get("name") and d.get("url")
            ]
            sets.append(DocSet(
                name=str(entry["name"]),
                corpus_dir=str(entry["corpus_dir"]),
                documents=docs,
            ))
        except (KeyError, TypeError):
            continue
    return sets


def _sha256_file(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _load_state(corpus_dir: str) -> dict:
    try:
        with open(os.path.join(corpus_dir, _STATE_FILENAME), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(corpus_dir: str, state: dict) -> None:
    path = os.path.join(corpus_dir, _STATE_FILENAME)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass


def refresh_document_set(doc_set: DocSet, force: bool = False) -> tuple[list[str], list[str], list[str]]:
    """Sync one set; rewrite a file only when its content changed.

    Returns ``(changed, unchanged, failed)`` filename lists. Per-document errors
    are captured (never raised) so one bad URL cannot abort the whole refresh."""
    import httpx  # lazy: only needed when actually refreshing

    os.makedirs(doc_set.corpus_dir, exist_ok=True)
    state = _load_state(doc_set.corpus_dir)
    changed: list[str] = []
    unchanged: list[str] = []
    failed: list[str] = []

    with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True, headers=_BROWSER_HEADERS) as client:
        for doc in doc_set.documents:
            dest = os.path.join(doc_set.corpus_dir, doc.name)
            entry = state.get(doc.name) if isinstance(state.get(doc.name), dict) else {}
            existing_hash = _sha256_file(dest) if os.path.exists(dest) else None

            request_headers: dict[str, str] = {}
            if not force and existing_hash is not None:
                if entry.get("etag"):
                    request_headers["If-None-Match"] = str(entry["etag"])
                if entry.get("last_modified"):
                    request_headers["If-Modified-Since"] = str(entry["last_modified"])

            try:
                resp = client.get(doc.url, headers=request_headers)
                if resp.status_code == 304:
                    unchanged.append(doc.name)
                    continue
                resp.raise_for_status()
                data = resp.content
                new_hash = hashlib.sha256(data).hexdigest()

                if not force and new_hash == existing_hash:
                    # Content identical: leave the file (and its mtime) untouched so
                    # the extraction cache stays valid; just refresh the validators.
                    entry = dict(entry)
                    entry.update({
                        "sha256": new_hash,
                        "etag": resp.headers.get("ETag", entry.get("etag")),
                        "last_modified": resp.headers.get("Last-Modified", entry.get("last_modified")),
                        "url": doc.url,
                    })
                    state[doc.name] = entry
                    unchanged.append(doc.name)
                    continue

                fd, tmp_path = tempfile.mkstemp(dir=doc_set.corpus_dir)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                os.replace(tmp_path, dest)
                state[doc.name] = {
                    "sha256": new_hash,
                    "etag": resp.headers.get("ETag"),
                    "last_modified": resp.headers.get("Last-Modified"),
                    "url": doc.url,
                    "bytes": len(data),
                }
                changed.append(doc.name)
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed.append(f"{doc.name} ({exc})")

    _save_state(doc_set.corpus_dir, state)
    return changed, unchanged, failed


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Refresh remote document corpora with sha256 change detection."
    )
    parser.add_argument("--config", default=None, help="path to the source recipe (default document_sources.json)")
    parser.add_argument("--set", dest="set_name", default=None, help="only refresh the named set")
    parser.add_argument("--force", action="store_true", help="re-download regardless of cached hash/etag")
    args = parser.parse_args(argv)

    sets = load_document_sources(args.config)
    if not sets:
        print("[docsync] No document sources found. Check document_sources.json.")
        return 1
    if args.set_name:
        sets = [s for s in sets if s.name == args.set_name]
        if not sets:
            print(f"[docsync] No set named '{args.set_name}'.")
            return 1

    total_changed = 0
    for doc_set in sets:
        changed, unchanged, failed = refresh_document_set(doc_set, force=args.force)
        total_changed += len(changed)
        print(
            f"[docsync] set '{doc_set.name}' ({doc_set.corpus_dir}): "
            f"{len(changed)} changed, {len(unchanged)} unchanged, {len(failed)} failed"
        )
        for name in changed:
            print(f"   CHANGED  {name}")
        for name in failed:
            print(f"   FAILED   {name}")

    if total_changed:
        print(
            f"[docsync] {total_changed} document(s) changed. Re-index to update Milvus "
            "(the manifest delta-sync re-embeds only the changed chunks)."
        )
    else:
        print("[docsync] No documents changed. The index is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
