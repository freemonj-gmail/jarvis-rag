# Minimal test helper commands

.PHONY: test smoke-test smoke-test-e2e

test: smoke-test

smoke-test:
	uv run python3 tests/smoke_test.py

# End-to-end smoke test: builds the full runtime (LLM + embeddings + Milvus)
# and runs real queries. Requires Ollama + Milvus running. Pass queries/corpus
# via ARGS, e.g. make smoke-test-e2e ARGS='"/path/to/corpus" "What jobs has X held?"'
smoke-test-e2e:
	uv run python3 tests/smoke_test_jarvis.py $(ARGS)
