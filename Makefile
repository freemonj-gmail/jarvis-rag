# Minimal test helper commands

.PHONY: test interrupt-test math-test smoke-test smoke-test-e2e mcp-login

test: interrupt-test math-test smoke-test

interrupt-test:
	uv run python tests/test_interrupt.py

math-test:
	uv run python tests/test_math.py

smoke-test:
	uv run python3 tests/smoke_test.py

# One-time OAuth sign-in for MCP connectors that require it (e.g. taxmcp.io).
# Opens your browser to authorize; caches tokens in jarvis_storage/.mcp_tokens.json
# (chmod 600, git-ignored) and refreshes them automatically afterward.
mcp-login:
	JARVIS_ENABLE_MCP=1 uv run python3 -m jarvisrag.mcp.connector

# End-to-end smoke test: builds the full runtime (LLM + embeddings + Milvus)
# and runs real queries. Requires Ollama + Milvus running. Pass queries/corpus
# via ARGS, e.g. make smoke-test-e2e ARGS='"/path/to/corpus" "What jobs has X held?"'
smoke-test-e2e:
	uv run python3 tests/smoke_test_jarvis.py $(ARGS)
