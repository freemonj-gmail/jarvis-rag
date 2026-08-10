# Minimal test helper commands

.PHONY: test interrupt-test math-test retrieval-eval-test answer-eval-test smoke-test smoke-test-e2e retrieval-eval answer-eval mcp-login

test: interrupt-test math-test retrieval-eval-test answer-eval-test smoke-test

interrupt-test:
	uv run python tests/test_interrupt.py

math-test:
	uv run python tests/test_math.py

retrieval-eval-test:
	uv run python tests/test_retrieval_evaluation.py

answer-eval-test:
	uv run python tests/test_answer_evaluation.py

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

# Retrieval-only metrics. Keep private cases/reports under jarvis_storage/evaluation/.
# Example: make retrieval-eval ARGS='"/path/to/corpus" "jarvis_storage/evaluation/cases.json" --persist-dir "jarvis_storage/evaluation/baseline" --output "jarvis_storage/evaluation/baseline.json"'
retrieval-eval:
	uv run python scripts/evaluate_retrieval.py $(ARGS)

# Answer-quality metrics through the REAL query pipeline (LLM + Milvus required).
# Slow and machine-intensive, so it is opt-in rather than part of `make test`.
# Use --repeat 3 when establishing a baseline to expose unstable cases.
# Example: make answer-eval ARGS='"/path/to/corpus" "jarvis_storage/evaluation/answer_cases.json" --case-details'
answer-eval:
	uv run python scripts/evaluate_answers.py $(ARGS)
