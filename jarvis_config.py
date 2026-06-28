"""Shared configuration, logging, and prompt templates for Jarvis.

Foundation module: it has no dependencies on other ``jarvis_*`` modules, so it
can be imported from anywhere without risking a circular import. Everything that
needs the debug flag, the module logger, the index-manifest constants, the
document-grounded QA prompt, or the active domain profile imports them from here.
"""
import logging
import os

from llama_index.core.prompts import PromptTemplate, PromptType

logger = logging.getLogger("jarvis")
INDEX_MANIFEST_FILE = "jarvis_index_manifest.json"
INDEX_MANIFEST_VERSION = 2

# Query, evidence, answer review, and runtime helper configuration.
JARVIS_DEBUG = os.environ.get("JARVIS_DEBUG", "0") == "1"


def debug_print(message: str) -> None:
    if JARVIS_DEBUG:
        print(message)


short_text_qa_template = PromptTemplate(
    "You are a document-grounded assistant.\n"
    "Use only the supplied context; do not use general model knowledge.\n"
    "If the context does not mention the requested names, files, entities, or facts, say that the indexed documents do not contain that information.\n"
    "Return partial findings only when they are directly relevant to the requested names, files, entities, or facts.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer in 2-6 concise bullet points when possible, with specific details (names, dates, amounts, IDs, addresses).",
    prompt_type=PromptType.QUESTION_ANSWER,
)

JARVIS_DOMAIN = os.environ.get("JARVIS_DOMAIN", "general").strip().lower()
if JARVIS_DOMAIN not in {"general", "tax"}:
    raise ValueError("JARVIS_DOMAIN must be 'general' or 'tax'.")

# ---------------------------------------------------------------------------
# MCP connector layer (opt-in). When enabled, Jarvis can pull fresh, authoritative
# intelligence (e.g. current tax authority via taxmcp.io) and personal/account
# data (e.g. brokerage data via Truthify) from external Model Context Protocol
# servers and inject it as highest-precedence evidence for institution-specific
# questions. Disabled by default; the registry file is pure data so new
# connectors drop in without code changes. See jarvis_mcp.py.
# ---------------------------------------------------------------------------
JARVIS_ENABLE_MCP = os.environ.get("JARVIS_ENABLE_MCP", "0") == "1"
JARVIS_MCP_CONFIG = os.environ.get("JARVIS_MCP_CONFIG", "jarvis_mcp.json")
JARVIS_MCP_TIMEOUT = float(os.environ.get("JARVIS_MCP_TIMEOUT", "20"))
# Where OAuth access/refresh tokens for MCP connectors are cached (file is
# written with 0600 perms; holds tokens only, never passwords). Connectors that
# authenticate via OAuth (e.g. taxmcp.io) do a one-time browser sign-in, then
# reuse/refresh the cached token from here on every later run.
JARVIS_MCP_TOKEN_CACHE = os.environ.get(
    "JARVIS_MCP_TOKEN_CACHE", os.path.join("jarvis_storage", ".mcp_tokens.json")
)
# Outbound PII guard for MCP requests. Only the user's question text ever leaves
# the machine (it is substituted into the MCP call arguments); local documents
# and retrieved evidence never egress. Before any MCP request fires, the
# outbound text is scanned for recognizable PII (SSN/EIN, email, phone, credit
# card, IP, street address). Modes:
#   "redact" (default) — replace detected PII with placeholders and warn, so the
#                         provider only ever sees a sanitized query.
#   "block"            — if any PII is detected, skip the MCP request entirely
#                         (nothing is sent) and answer on local evidence only.
#   "off"              — send the query verbatim (no scanning).
JARVIS_MCP_PII_GUARD = os.environ.get("JARVIS_MCP_PII_GUARD", "redact").strip().lower()

