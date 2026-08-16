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
# Empty default => use the registry.json packaged inside jarvisrag/mcp/. Set this
# env var to point at a custom registry file (absolute or relative to the cwd).
JARVIS_MCP_CONFIG = os.environ.get("JARVIS_MCP_CONFIG", "")
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

# ---------------------------------------------------------------------------
# Confidence-routed page extraction pipeline (OCR/scanned-doc enhancement,
# Phase 1). Enabled by default: scanned PDFs and image files flow through
# jarvisrag/extraction/pagepipe.py, which:
#   1. prefers the native PDF text layer (PyMuPDF) with proper block reading
#      order whenever a page actually contains extractable text, and
#   2. for image-only / sparse pages, transcribes text with Apple Vision OCR on
#      macOS (see JARVIS_ENABLE_APPLE_OCR) — or, when that is unavailable, uses
#      Surya for layout + per-line recognition confidence, routing ONLY the
#      low-confidence regions to the pytesseract/easyocr fallback.
# Every stage records per-region provenance + confidence in Document metadata.
# The whole feature degrades gracefully: if PyMuPDF/Apple Vision/Surya are not
# installed, the extractors fall back to their original behavior.
# ---------------------------------------------------------------------------
# Always on: the confidence-routed page pipeline is the native extraction path.
JARVIS_ENABLE_PAGE_PIPELINE = True
# Enable the Surya layout + recognition stage specifically. It requires the
# optional ``surya-ocr`` dependency (see the ``ocr`` extra in pyproject.toml);
# when off (or Surya is unavailable) the pipeline still uses the native PDF text
# layer and the pytesseract/easyocr fallback.
JARVIS_ENABLE_SURYA = os.environ.get("JARVIS_ENABLE_SURYA", "0") == "1"
# A recognition region whose confidence is at or above this threshold is trusted
# as-is; anything below it is re-OCR'd with the pytesseract/easyocr fallback and
# the higher-confidence result is kept. Range 0.0-1.0.
JARVIS_OCR_MIN_CONF = float(os.environ.get("JARVIS_OCR_MIN_CONF", "0.65"))
# Minimum characters of extractable native text a PDF page must yield before it
# is treated as a real (born-digital) text layer. Below this the page is deemed
# image-only/scanned and is escalated to the OCR path.
JARVIS_PDF_MIN_CHARS_PER_PAGE = int(os.environ.get("JARVIS_PDF_MIN_CHARS_PER_PAGE", "40"))
# Document-level OCR quality validator + escalation. When a whole scanned page's
# assembled text fails a vocabulary-grounded garbage/empty check, the pipeline
# does NOT blindly keep that first result: it escalates to the other enabled
# local engines (the VLM when on, then the pytesseract/easyocr fallback) and
# keeps the highest-quality, non-garbage transcription. This closes the
# "confidently-wrong or empty page slips through" gap using only local engines
# already present — no new dependency, no cloud, no fee. Always on.
JARVIS_OCR_ESCALATE_ON_FAILURE = True
# Resolution (DPI) used to rasterize a PDF page when it must be sent through the
# OCR path. 300 DPI is the accuracy/latency sweet spot for document OCR.
JARVIS_PDF_OCR_DPI = int(os.environ.get("JARVIS_PDF_OCR_DPI", "300"))

# ---------------------------------------------------------------------------
# Vision-language model (VLM) escalation for the page pipeline (OCR/scanned-doc
# enhancement, Phase 2). Opt-in and disabled by default. When enabled (and the
# ``transformers`` stack is installed), regions the cheaper OCR stages could not
# read confidently — and, when Surya is unavailable, whole scanned pages — are
# escalated to a local vision-language model (default SmolVLM2 by Hugging Face,
# Apache-2.0-licensed) for a second-opinion OCR. Optionally, figure/photo
# regions can be captioned so non-text visual content becomes searchable.
# Runs at index time on region crops, is cached, and degrades gracefully: any
# load/inference failure falls back to the pytesseract/easyocr result.
# ---------------------------------------------------------------------------
JARVIS_ENABLE_VLM = os.environ.get("JARVIS_ENABLE_VLM", "0") == "1"
# Hugging Face model id for the VLM. Defaults to SmolVLM2-2.2B-Instruct (Hugging
# Face, Apache-2.0): a modern-transformers-native image-text-to-text model that
# runs on Apple Silicon without any frozen remote code. Set
# "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" for a lighter/faster (lower
# fidelity) run, or point at any other VLM transformers supports.
JARVIS_VLM_MODEL = os.environ.get("JARVIS_VLM_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct").strip()
# Confidence assigned to VLM-recovered text. The VLM exposes no numeric
# confidence, so this fixed value marks the span as trusted (above the OCR
# threshold) while remaining distinguishable from native text (1.0) for later
# confidence-aware ranking. Range 0.0-1.0.
JARVIS_VLM_CONF = float(os.environ.get("JARVIS_VLM_CONF", "0.75"))
# When enabled, also caption figure/photo regions (and whole image-only pages)
# with the VLM so non-text visual content (IDs, charts, signatures) becomes
# searchable. Off by default to avoid adding descriptive noise to text corpora.
JARVIS_VLM_CAPTION = os.environ.get("JARVIS_VLM_CAPTION", "0") == "1"

# ---------------------------------------------------------------------------
# Apple Vision OCR backend for the page pipeline (on-device text recognition).
# Enabled by default on macOS; a no-op elsewhere. When the ``ocrmac`` wrapper
# around Apple's Vision framework is available, scanned/photographed pages and
# low-confidence regions are transcribed with Apple Vision *ahead of* the VLM on
# the text path. Unlike a general VLM, Vision OCR is verbatim (it does not
# paraphrase or fabricate field values), hardware-accelerated (sub-second per
# image, no timeouts), and free — the preferred text engine for documents such
# as IDs, receipts, and forms; the VLM is reserved for captioning visual content
# (JARVIS_VLM_CAPTION). Requires the page pipeline (JARVIS_ENABLE_PAGE_PIPELINE,
# also on by default). It self-disables on non-macOS platforms (and on any
# import/inference failure), falling back to the Surya/pytesseract/easyocr path,
# so non-Apple users get the legacy OCR behavior automatically. Set
# JARVIS_ENABLE_APPLE_OCR=0 to force it off.
# ---------------------------------------------------------------------------
JARVIS_ENABLE_APPLE_OCR = os.environ.get("JARVIS_ENABLE_APPLE_OCR", "1") == "1"
# Comma-separated Vision language preferences for post-processing/recognition,
# e.g. "en-US" or "en-US,de-DE". See the IANA language-subtag registry for codes.
JARVIS_APPLE_OCR_LANGS = os.environ.get("JARVIS_APPLE_OCR_LANGS", "en-US").strip()
# Vision recognition level: "accurate" (default, best fidelity) or "fast".
JARVIS_APPLE_OCR_LEVEL = os.environ.get("JARVIS_APPLE_OCR_LEVEL", "accurate").strip()

