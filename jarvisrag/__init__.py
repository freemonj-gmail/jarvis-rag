"""Jarvis RAG — local document Retrieval-Augmented Generation assistant.

Source is organized into function subpackages:

- ``config``      runtime configuration + logging
- ``iotools``     process/stdout/timeout/warning-filter helpers
- ``text``        text processing (``textproc``) + query analysis (``query``)
- ``extraction``  document extractors + indexing
- ``vectorstore`` Milvus / local vector store
- ``retrieval``   hybrid retriever + evidence pipeline
- ``answer``      answer synthesis (``answer``) + startup wiring (``bootstrap``)
- ``speech``      text-to-speech
- ``mcp``         external Model Context Protocol connector layer (``connector``)

The thin entry point ``Jarvis.py`` and the backward-compatible facade
``jarvis_helpers.py`` remain at the repository root.
"""
