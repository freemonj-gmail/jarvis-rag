# Jarvis RAG

A local document Retrieval-Augmented Generation (RAG) assistant with multi-format
parsing, OCR fallbacks, a Milvus / local vector store, and optional text-to-speech.

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management
- (Optional) A running [Milvus](https://milvus.io/) server at `127.0.0.1:19530`
  for the default vector backend
- (Optional) [Ollama](https://ollama.com/) for the local LLM

## Setup

```bash
uv sync
```

## Run

```bash
# Point Jarvis at a folder of documents (PDF, DOCX, XLSX, PPTX, images, etc.)
uv run python Jarvis.py /path/to/your/documents
```

## Tests

```bash
# Fast helper smoke test (no heavy model downloads)
make smoke-test
# or
uv run python tests/smoke_test.py

# Full Jarvis init smoke test (reuses extraction cache + index)
uv run python smoke_test_jarvis.py "What is in my documents?"

# Live Milvus validation (requires a running Milvus server)
uv run python tests/milvus_live_test.py
```

## Key environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `JARVIS_VECTOR_BACKEND` | `milvus` | `milvus`, `local`, or `memory` |
| `JARVIS_REBUILD_INDEX` | `0` | Force a full re-embed of the corpus |
| `JARVIS_CONTEXT_MODE` | `chunk` | `chunk` or `source_window` |
| `JARVIS_DOMAIN` | `general` | `general` or `tax` detail profile |
| `JARVIS_DISABLE_TTS` | unset | Set to `1` to disable text-to-speech |
| `USE_MILVUS` | `1` | Toggle the Milvus backend |
| `JARVIS_ROUTER` | `0` | Set to `1` to enable the two-tier LLM router (escalate out-of-corpus questions to a larger model) |
| `JARVIS_ESCALATION_MODEL` | `llama3:70b` | Ollama tag for the escalation model used when the router is enabled |
| `JARVIS_ESCALATION_TIMEOUT` | `600` | Request timeout (seconds) for the escalation model |
| `JARVIS_ESCALATION_TEMPERATURE` | `0.2` | Sampling temperature for the escalation model |

## Layout

```
Jarvis.py            # Application entry point
jarvis_helpers.py    # Parsers, OCR, vector store, RAG pipeline, TTS
tests/               # Smoke + Milvus tests
jarvis_storage/      # Generated extraction cache + persistent index (gitignored)
```
