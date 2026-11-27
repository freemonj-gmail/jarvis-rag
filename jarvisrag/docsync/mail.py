"""Apple Mail corpus staging: selected .emlx messages and their attachments.

Messages are staged into the corpus by Message-ID, scoped by account and
mailbox, and each message's attachments are mirrored beside the corpus.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from typing import Optional

from jarvisrag.docsync.sources import _load_state, _save_state, _sha256_file, DocSet


def _msgid_key(identity: str) -> str:
    """Flat, stable corpus filename for one LOGICAL message.

    Keyed by RFC Message-ID so Gmail's copies of the same email (stored once per
    label / All Mail as separate files) collapse to a single corpus document.
    Flattened via hash because the domain corpus loader reads only the top level
    of ``corpus_dir`` (non-recursive)."""
    return hashlib.sha1(identity.encode("utf-8", "surrogatepass")).hexdigest()[:24] + ".emlx"


def mail_attachment_dir(corpus_dir: str) -> str:
    """Where a mail corpus keeps its attachment mirror.

    A SIBLING of ``corpus_dir``, never inside it: the domain loader reads the
    corpus recursively, so a nested directory would be ingested into the index.
    Attachments are dereferenced at answer time only; nothing here is embedded."""
    return os.path.normpath(corpus_dir) + "_attachments"


def _clone_or_copy(src: str, dst: str) -> None:
    """Copy ``src`` to ``dst``, as an APFS clone when possible.

    A clone shares the source's storage blocks, so mirroring Apple Mail's
    attachment files costs almost no extra disk on the same volume. Falls back
    to a regular copy on any other filesystem or platform."""
    tmp = dst + ".mirror-tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    cloned = False
    try:
        import ctypes
        import ctypes.util
        import sys

        if sys.platform == "darwin":
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            cloned = libc.clonefile(os.fsencode(src), os.fsencode(tmp), 0) == 0
    except Exception:  # noqa: BLE001 - any clone failure falls back to a copy
        cloned = False
    if not cloned:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _mirror_message_attachments(src: str, dest_dir: str) -> bool:
    """Mirror one Apple Mail message's externally stored attachment files.

    Apple Mail keeps a message's attachments OUTSIDE its ``.emlx`` (leaving only
    the MIME headers), under ``<...>/Attachments/<message number>/<part>/<file>``
    beside the ``Messages`` directory. Mirroring them here lets a Jarvis process
    WITHOUT Full Disk Access open an attachment later, on demand. Files already
    mirrored at the same size are skipped. Returns True when the message has an
    attachment directory."""
    stem = os.path.basename(src).split(".", 1)[0]
    source_dir = os.path.join(os.path.dirname(os.path.dirname(src)), "Attachments", stem)
    if not os.path.isdir(source_dir):
        return False
    for root, _dirs, files in os.walk(source_dir):
        for fname in files:
            if fname.startswith("."):
                continue
            source_file = os.path.join(root, fname)
            target = os.path.join(dest_dir, os.path.relpath(source_file, source_dir))
            try:
                if os.path.exists(target) and os.path.getsize(target) == os.path.getsize(source_file):
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                _clone_or_copy(source_file, target)
            except OSError:
                continue  # one unreadable attachment must not stop the refresh
    return True


def _sync_attachment_mirror(
    doc_set: "DocSet", sources: dict, changed_ids: set, force: bool, prune: bool
) -> None:
    """Keep the attachment mirror aligned with the staged messages.

    Each LOGICAL message (one staged ``.emlx``) is mirrored once, from the first
    source copy that has an attachment directory; a changed message is
    re-mirrored. Mirrors whose message is no longer staged are removed, but only
    after a complete scan so an interrupted refresh never deletes attachments.
    Nothing here is extracted or embedded."""
    mirror_root = mail_attachment_dir(doc_set.corpus_dir)
    staged: set[str] = set()
    done: set[str] = set()
    for src, entry in sources.items():
        identity = entry.get("id") if isinstance(entry, dict) else None
        if not identity:
            continue
        stem = _msgid_key(str(identity))[: -len(".emlx")]
        staged.add(stem)
        if stem in done:
            continue
        target = os.path.join(mirror_root, stem)
        if not force and identity not in changed_ids and os.path.isdir(target):
            done.add(stem)
            continue
        if _mirror_message_attachments(src, target):
            done.add(stem)
    if not prune or not os.path.isdir(mirror_root):
        return
    with os.scandir(mirror_root) as entries:
        stale = [entry.path for entry in entries if entry.is_dir() and entry.name not in staged]
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)


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

    _sync_attachment_mirror(doc_set, new_sources, set(changed_ids), force, prune=not failed)

    if state.get("_sources") != new_sources:
        state["_sources"] = new_sources
        if not _save_state(doc_set.corpus_dir, state):
            failed.append("Could not save mail refresh state; the next refresh will retry.")
    return changed, unchanged, failed
