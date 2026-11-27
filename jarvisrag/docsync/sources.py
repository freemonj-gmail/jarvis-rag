"""Remote document sources: the recipe, per-file fetch state, and downloads.

A corpus file is rewritten only when its fetched bytes changed, so unchanged
files keep their mtime and their extraction cache.
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
    """A named group of remote documents synced into one corpus directory.

    ``match_any``/``label`` are optional routing metadata consumed by the
    domain-corpus layer (not by docsync itself)."""
    name: str
    corpus_dir: str
    documents: list[DocSource] = field(default_factory=list)
    match_any: list[str] = field(default_factory=list)
    label: str = ""
    type: str = "remote"          # "remote" (URL documents) | "mail" (local Apple Mail store)
    mail_root: str = ""           # mail sets: path to the Apple Mail store (default ~/Library/Mail)
    exclude_folders: list[str] = field(default_factory=list)  # mail sets: mailbox names to skip
    # Account identifiers are machine-specific, so they come from the environment and are
    # never committed in the recipe. Empty = index every account.
    include_accounts: list[str] = field(default_factory=list)


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
                match_any=[str(t) for t in entry.get("match_any", [])],
                label=str(entry.get("label", "")),
                type=str(entry.get("type", "remote")),
                mail_root=str(entry.get("mail_root", "")),
                exclude_folders=[str(t) for t in entry.get("exclude_folders", [])],
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


def _save_state(corpus_dir: str, state: dict) -> bool:
    path = os.path.join(corpus_dir, _STATE_FILENAME)
    tmp = None
    try:
        descriptor, tmp = tempfile.mkstemp(dir=corpus_dir, prefix=".docsync-")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return True
    except OSError:
        return False
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
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

    if not _save_state(doc_set.corpus_dir, state):
        failed.append("Could not save refresh state; the next refresh may repeat work.")
    return changed, unchanged, failed
