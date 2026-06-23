"""Smoke test for Jarvis.py that reuses ALL of Jarvis's caches:
  * Per-file extraction cache (jarvis_storage/extraction_cache/) — re-extracts
    a PDF/DOCX only when its content hash changes.
  * Persistent index manifest (jarvis_storage/jarvis_index_manifest.json) —
    skips re-embedding into Milvus when the document signature matches.

It does this by importing Jarvis directly so its module-level init runs once.
TTS is stubbed out via JARVIS_DISABLE_TTS=1.

Usage:
    JARVIS_DEBUG=1 uv run python smoke_test_jarvis.py "What is the definition of weather?"
    uv run python smoke_test_jarvis.py "Q one" "Q two"          # chained turns
    uv run python smoke_test_jarvis.py /path/to/corpus "Q one"  # override input_dir
"""
import os
import sys

# Stub TTS BEFORE importing Jarvis (its module-level init reads this env var).
os.environ.setdefault("JARVIS_DISABLE_TTS", "1")

# If the first arg is an existing directory, treat it as the input_dir
# override that Jarvis.py reads from sys.argv[1]. Otherwise leave Jarvis to
# use its default. Either way the extraction cache + Milvus manifest are
# reused — Jarvis only re-extracts files whose hash changed and only
# re-embeds chunks not already present in the persistent index.
args = sys.argv[1:]
if args and os.path.isdir(args[0]):
    sys.argv = [sys.argv[0], args[0]]
    queries = args[1:]
else:
    sys.argv = [sys.argv[0]]
    queries = args

if not queries:
    queries = ["What is the definition of weather?"]

print("=" * 70)
print(f"[smoke] Importing Jarvis (init will reuse extraction cache + Milvus index)...")
print(f"[smoke] Queries to run: {queries}")
print("=" * 70)

import Jarvis  # noqa: E402  -- triggers full Jarvis init exactly once

print("=" * 70)
print(f"[smoke] Jarvis ready. Running {len(queries)} query/queries.")
print("=" * 70)

last_turn = {}
for i, q in enumerate(queries, start=1):
    print(f"\n[smoke] === Query {i}/{len(queries)} ===")
    print(f"[smoke] You: {q}")
    last_turn = Jarvis.handle_query(q, Jarvis.runtime, last_turn)

print("\n[smoke] All queries complete.")
