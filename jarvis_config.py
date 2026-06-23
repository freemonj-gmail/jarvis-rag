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
