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
import shutil
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


def _msgid_key(identity: str) -> str:
    """Flat, stable corpus filename for one LOGICAL message.

    Keyed by RFC Message-ID so Gmail's copies of the same email (stored once per
    label / All Mail as separate files) collapse to a single corpus document.
    Flattened via hash because the domain corpus loader reads only the top level
    of ``corpus_dir`` (non-recursive)."""
    return hashlib.sha1(identity.encode("utf-8", "surrogatepass")).hexdigest()[:24] + ".emlx"


def _emlx_message_id(src: str) -> Optional[str]:
    """Best-effort RFC Message-ID from an ``.emlx`` file (stdlib only).

    Reads a bounded header prefix and strips the ``.emlx`` byte-count framing
    line before parsing. Returns None when absent/unreadable so the caller can
    fall back to a path identity (that message is kept, not merged)."""
    import email
    from email import policy

    try:
        with open(src, "rb") as handle:
            chunk = handle.read(65536)
    except OSError:
        return None
    newline = chunk.find(b"\n")
    header_bytes = chunk[newline + 1:] if newline != -1 else chunk
    try:
        message = email.message_from_bytes(header_bytes, policy=policy.default)
        mid = message.get("Message-ID") or message.get("Message-Id")
    except Exception:
        return None
    mid = str(mid).strip() if mid else ""
    return mid or None


def _mail_identity(mail_root: str, src: str, msgid: Optional[str]) -> str:
    """Dedup identity for a message: its Message-ID, else its source path."""
    return msgid if msgid else "path:" + os.path.relpath(src, mail_root)


def _normalized_mailbox_names(names: list[str]) -> set[str]:
    return {n.strip().lower() for n in names if n and n.strip()}


def _is_excluded_segment(segment: str, wanted: set[str]) -> bool:
    """Apple Mail nests messages under per-mailbox ``.mbox`` bundles."""
    return segment.lower().removesuffix(".mbox") in wanted


def _excluded_mailbox(relative_path: str, wanted: set[str]) -> bool:
    """True when any directory of ``relative_path`` is an excluded mailbox.

    Compares whole path SEGMENTS so ``Trash`` never matches a user folder that
    merely contains the word."""
    if not wanted:
        return False
    segments = os.path.dirname(relative_path).split(os.sep)
    return any(
        _is_excluded_segment(segment, wanted)
        for segment in segments if segment not in ("", ".", "..")
    )


def _configured_accounts(doc_set: "DocSet") -> set[str]:
    """Accounts to index, from the set or ``JARVIS_MAIL_ACCOUNTS`` (empty = all)."""
    configured = doc_set.include_accounts or os.environ.get("JARVIS_MAIL_ACCOUNTS", "").split(",")
    return _normalized_mailbox_names(list(configured))


def _included_account(relative_path: str, wanted: set[str]) -> bool:
    """True when ``relative_path`` sits under one of the configured accounts."""
    if not wanted:
        return True
    return any(segment.lower() in wanted for segment in relative_path.split(os.sep))


def refresh_mail_set(doc_set: DocSet, force: bool = False) -> tuple[list[str], list[str], list[str]]:
    """Sync a local Apple Mail store into a DE-DUPLICATED corpus of ``.emlx``.

    Safety by construction: reads ONLY individual message files, never Mail's
    SQLite Envelope Index, so Mail is never locked or blocked. A stat-only walk
    finds changes against saved per-source state. Changed files are copied into
    ``corpus_dir`` before parsing their Message-ID. Identical copies leave the
    destination mtime untouched so extraction and indexing can reuse their caches.

    De-duplication is ALWAYS on and driven here (no caller flag): the corpus
    filename is keyed by RFC Message-ID, so Gmail's per-label copies of one email
    collapse to a single document; a message with no Message-ID falls back to a
    path identity (kept, never merged). A corpus document is pruned only when NO
    remaining source file references its Message-ID. Only deltas are read after
    the first run; nothing is ever written back to the mail store. Returns
    ``(changed, unchanged, failed)`` where ``changed`` counts LOGICAL messages
    and includes ``removed:<id>`` entries so a deletion still triggers re-index.

    Requires Full Disk Access for the process to read ~/Library/Mail."""
    mail_root = os.path.expanduser(doc_set.mail_root or "~/Library/Mail")
    os.makedirs(doc_set.corpus_dir, exist_ok=True)
    state = _load_state(doc_set.corpus_dir)
    prior = state.get("_sources") if isinstance(state.get("_sources"), dict) else None
    changed: list[str] = []
    unchanged: list[str] = []
    failed: list[str] = []

    if not os.path.isdir(mail_root):
        return changed, unchanged, [f"mail_root not found: {mail_root}"]

    current: dict[str, dict] = {}

    def scan_error(error):
        failed.append(f"Incomplete mail scan ({type(error).__name__}); refresh skipped.")

    # A message kept only in a skipped account/mailbox loses its last source here, so the
    # existing prune drops its staged copy; one also filed elsewhere keeps that path.
    excluded_names = _normalized_mailbox_names(doc_set.exclude_folders)
    account_names = _configured_accounts(doc_set)
    for root, dirs, files in os.walk(mail_root, onerror=scan_error):
        if excluded_names:
            dirs[:] = [d for d in dirs if not _is_excluded_segment(d, excluded_names)]
        for fname in files:
            if not fname.lower().endswith(".emlx"):
                continue
            src = os.path.join(root, fname)
            relative = os.path.relpath(src, mail_root)
            if not _included_account(relative, account_names):
                continue
            if _excluded_mailbox(relative, excluded_names):
                continue
            try:
                st = os.stat(src)
            except OSError as exc:
                scan_error(exc)
                continue
            current[src] = {"mtime": int(st.st_mtime), "mtime_ns": st.st_mtime_ns, "size": int(st.st_size)}

    if failed:
        return changed, unchanged, failed

    # Guard: an empty scan while anything already exists almost always means the
    # process lacks Full Disk Access; never mass-drop or wipe the corpus on that.
    with os.scandir(doc_set.corpus_dir) as entries:
        corpus_has_files = any(entry.name.endswith(".emlx") for entry in entries)
    if not current and (prior or corpus_has_files or state.get("_messages")):
        return changed, unchanged, [
            "mail scan found no messages (Full Disk Access not granted?) — "
            "skipping to avoid dropping the corpus"
        ]

    prior = prior or {}
    new_sources: dict[str, dict] = {}
    copied_ids: set[str] = set()
    changed_ids: dict[str, None] = {}
    unchanged_ids: dict[str, None] = {}

    for src, info in current.items():
        was = prior.get(src) if isinstance(prior.get(src), dict) else None
        if (not force and was and was.get("mtime_ns") == info["mtime_ns"]
                and was.get("size") == info["size"] and was.get("id")):
            identity = str(was["id"])
            dest = os.path.join(doc_set.corpus_dir, _msgid_key(identity))
            if os.path.exists(dest):
                new_sources[src] = dict(info, id=identity)
                unchanged_ids[identity] = None
                continue
        tmp_dest = None
        try:
            descriptor, tmp_dest = tempfile.mkstemp(dir=doc_set.corpus_dir, prefix=".mail-")
            os.close(descriptor)
            shutil.copyfile(src, tmp_dest)
            after = os.stat(src)
            if (after.st_mtime_ns != info["mtime_ns"] or after.st_size != info["size"]
                    or os.stat(tmp_dest).st_size != info["size"]):
                raise OSError("Source changed while copying; retry on next refresh.")
            identity = _mail_identity(mail_root, src, _emlx_message_id(tmp_dest))
            dest = os.path.join(doc_set.corpus_dir, _msgid_key(identity))
            if identity not in copied_ids:
                copied_hash = _sha256_file(tmp_dest)
                if copied_hash is None:
                    raise OSError("Could not validate staged copy.")
                if not force and copied_hash == _sha256_file(dest):
                    unchanged_ids[identity] = None
                else:
                    os.replace(tmp_dest, dest)
                    changed_ids[identity] = None
                copied_ids.add(identity)
            new_sources[src] = dict(info, id=identity)
        except Exception as exc:  # noqa: BLE001 - report and continue
            if was:
                new_sources[src] = was
            # Class only: the exception text can embed message content.
            failed.append(f"{os.path.basename(src)} ({type(exc).__name__})")
        finally:
            if tmp_dest and os.path.exists(tmp_dest):
                try:
                    os.remove(tmp_dest)
                except OSError:
                    pass

    changed = list(changed_ids)
    unchanged = [identity for identity in unchanged_ids if identity not in changed_ids]
    if failed:
        for src, entry in prior.items():
            new_sources.setdefault(src, entry)

    current_ids = {v["id"] for v in new_sources.values()}
    prior_ids = {v.get("id") for v in prior.values() if isinstance(v, dict)}
    for identity in prior_ids - current_ids:
        if not identity:
            continue
        gone = os.path.join(doc_set.corpus_dir, _msgid_key(str(identity)))
        try:
            if os.path.exists(gone):
                os.remove(gone)
        except OSError as exc:
            failed.append(f"Could not remove staged message ({type(exc).__name__}).")
            for src, entry in prior.items():
                if isinstance(entry, dict) and entry.get("id") == identity:
                    new_sources.setdefault(src, entry)
            continue
        changed.append(f"removed:{identity}")

    if state.get("_sources") != new_sources:
        state["_sources"] = new_sources
        if not _save_state(doc_set.corpus_dir, state):
            failed.append("Could not save mail refresh state; the next refresh will retry.")
    return changed, unchanged, failed


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Refresh remote document corpora with sha256 change detection."
    )
    parser.add_argument("--config", default=None, help="path to the source recipe (default document_sources.json)")
    parser.add_argument("--set", dest="set_name", default=None, help="only refresh the named set")
    parser.add_argument(
        "--type", dest="set_type", default=None,
        help="only refresh sets of this type ('remote' covers every downloaded corpus and needs no "
             "Full Disk Access; 'mail' syncs the local Apple Mail store and does)",
    )
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
    if args.set_type:
        sets = [s for s in sets if s.type == args.set_type]
        if not sets:
            print(f"[docsync] No sets of type '{args.set_type}'.")
            return 1

    from jarvisrag.iotools import Spinner  # lazy: only the CLI run needs it

    total_changed = 0
    total_failed = 0
    max_list = 20  # cap the per-file listing so a large mail delta doesn't flood the terminal
    for doc_set in sets:
        with Spinner(f"Refreshing {doc_set.name}"):
            if doc_set.type == "mail":
                changed, unchanged, failed = refresh_mail_set(doc_set, force=args.force)
            else:
                changed, unchanged, failed = refresh_document_set(doc_set, force=args.force)
        total_changed += len(changed)
        total_failed += len(failed)
        print(
            f"[docsync] set '{doc_set.name}' ({doc_set.corpus_dir}): "
            f"{len(changed)} changed, {len(unchanged)} unchanged, {len(failed)} failed"
        )
        for name in changed[:max_list]:
            print(f"   CHANGED  {name}")
        if len(changed) > max_list:
            print(f"   … and {len(changed) - max_list} more changed")
        for name in failed[:max_list]:
            print(f"   FAILED   {name}")
        if len(failed) > max_list:
            print(f"   … and {len(failed) - max_list} more failed")

    if total_changed:
        print(
            f"[docsync] {total_changed} document(s) changed. Re-index to update Milvus "
            "(the manifest delta-sync re-embeds only the changed chunks)."
        )
    elif not total_failed:
        print("[docsync] No source documents changed. Indexing is a separate step on Jarvis startup.")
    if total_failed:
        print("[docsync] Refresh incomplete; failed items will be retried on the next refresh.")
    print("[docsync] Refresh complete.")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
