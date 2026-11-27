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

Split by job: ``sources`` (the recipe and remote downloads) and ``mail``
(Apple Mail staging); this module is the ``python -m jarvisrag.docsync`` command.
"""

from __future__ import annotations

from typing import Optional

from jarvisrag.docsync.sources import load_document_sources, refresh_document_set
from jarvisrag.docsync.mail import refresh_mail_set


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
