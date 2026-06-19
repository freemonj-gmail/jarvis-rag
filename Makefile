# Minimal test helper commands

.PHONY: test smoke-test

test: smoke-test

smoke-test:
	uv run python3 tests/smoke_test.py
