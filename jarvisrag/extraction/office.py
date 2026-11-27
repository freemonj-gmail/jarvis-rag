"""Office, archive and web-page readers.

ZIP archives (each member read by its own extension's reader), Excel,
PowerPoint, Docling/Unstructured office parsing, saved MHTML pages, and the
deferred wrapper that builds a heavy reader on first use.
"""

import os
import tempfile
import zipfile
from typing import List, Optional

import pandas as pd
from llama_index.core.schema import Document
from llama_index.readers.file.unstructured import UnstructuredReader
from pptx import Presentation

from jarvisrag.config import logger
from jarvisrag.extraction.indexing import chunk_text
from jarvisrag.text.textproc import normalize_text


class ZipExtractor:
    def __init__(self, file_extractors):
        self.file_extractors = file_extractors

    def load_data(self, file_path, **kwargs):
        docs = []
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_ref.extractall(tmpdirname)
                for root, _, files in os.walk(tmpdirname):
                    for name in files:
                        ext = os.path.splitext(name)[1].lower()
                        extractor = self.file_extractors.get(ext)
                        if extractor:
                            extracted_docs = extractor.load_data(os.path.join(root, name), **kwargs)
                            docs.extend(extracted_docs)
        return docs


class ExcelReader:
    def load_data(self, file_path, **kwargs):
        docs = []
        file_name = os.path.basename(file_path)
        try:
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                text = normalize_text(df.to_string(index=False))
                if not text:
                    continue
                for chunk_index, chunk in enumerate(chunk_text(text)):
                    docs.append(Document(text=chunk, metadata={
                        "file_path": str(file_path),
                        "file_name": file_name,
                        "sheet": sheet_name,
                        "chunk_index": chunk_index,
                        "extracted_by": "excel"
                    }))
        except Exception as e:
            logger.error(f"Failed to read Excel file {file_path}: {e}")
        return docs


class PowerPointReader:
    def load_data(self, file_path, **kwargs):
        docs = []
        file_name = os.path.basename(file_path)
        try:
            prs = Presentation(file_path)
            for slide_index, slide in enumerate(prs.slides, start=1):
                slide_text = ""
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        slide_text += shape.text + "\n"
                slide_text = normalize_text(slide_text)
                if not slide_text:
                    continue
                for chunk_index, chunk in enumerate(chunk_text(slide_text)):
                    docs.append(Document(text=chunk, metadata={
                        "file_path": str(file_path),
                        "file_name": file_name,
                        "slide_index": slide_index,
                        "chunk_index": chunk_index,
                        "extracted_by": "powerpoint"
                    }))
        except Exception as e:
            logger.error(f"Failed to read PowerPoint file {file_path}: {e}")
        return docs


class DeferredExtractor:
    def __init__(self, factory):
        self._factory = factory
        self._extractor = None

    def load_data(self, *args, **kwargs):
        if self._extractor is None:
            self._extractor = self._factory()
        return self._extractor.load_data(*args, **kwargs)


class DoclingReader:
    """Parse modern Office / structured documents via Docling and emit markdown
    chunks suitable for vector embeddings. Supports the Docling-native formats
    (.docx, .pptx, .xlsx, .html, .md, .csv, etc.); legacy binary formats
    (.doc/.xls/.ppt) and .rtf must be routed to a different reader. Falls back
    to UnstructuredReader when docling can't recognize the file (e.g. a legacy
    .doc renamed with a .docx extension)."""

    _converter = None
    _fallback = None

    @classmethod
    def _get_converter(cls):
        if cls._converter is None:
            from docling.document_converter import DocumentConverter
            cls._converter = DocumentConverter()
        return cls._converter

    @classmethod
    def _get_fallback(cls):
        if cls._fallback is None:
            cls._fallback = UnstructuredReader()
        return cls._fallback

    def _fallback_load(self, file_path_str, file_name, reason):
        logger.info(
            f"Docling could not parse {file_name} ({reason}); falling back to UnstructuredReader."
        )
        try:
            fb_docs = self._get_fallback().load_data(file_path_str) or []
        except Exception as e:
            logger.error(f"UnstructuredReader fallback failed for {file_path_str}: {e}")
            return []
        out: List[Document] = []
        for fb in fb_docs:
            text = normalize_text(getattr(fb, "text", "") or "")
            if not text:
                continue
            for chunk_index, chunk in enumerate(
                chunk_text(text, chunk_size=800, chunk_overlap=160)
            ):
                out.append(Document(text=chunk, metadata={
                    "file_path": file_path_str,
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "extracted_by": "unstructured_fallback",
                }))
        return out

    def load_data(self, file_path, **kwargs):
        file_path_str = str(file_path)
        file_name = os.path.basename(file_path_str)
        try:
            result = self._get_converter().convert(file_path_str)
            text = normalize_text(result.document.export_to_markdown())
        except Exception as e:
            return self._fallback_load(file_path_str, file_name, str(e))
        if not text:
            return self._fallback_load(file_path_str, file_name, "empty docling output")
        docs: List[Document] = []
        # ~800-word chunks with 160-word overlap fits comfortably in bge-m3's
        # 8192-token context window (~6x headroom) while keeping per-section
        # semantic focus. Going larger would dilute retrieval ranking; going
        # smaller would split section headers from their items. Send bigger
        # context to the LLM downstream via JARVIS_SOURCE_HINT_CHUNK_CHARS,
        # JARVIS_MAX_EVIDENCE_LINES, and JARVIS_RETRIEVAL_TOP_K instead.
        for chunk_index, chunk in enumerate(
            chunk_text(text, chunk_size=800, chunk_overlap=160)
        ):
            docs.append(Document(text=chunk, metadata={
                "file_path": file_path_str,
                "file_name": file_name,
                "chunk_index": chunk_index,
                "extracted_by": "docling",
            }))
        return docs


class MHTMLReader:
    """Parse .mhtml/.mht MIME web-archive files (single-file "webpage, complete"
    saves). The container is a MIME multipart message; extract its primary
    text/html part with the stdlib ``email`` parser, then delegate the HTML to
    ``html_reader`` (Docling, with its UnstructuredReader fallback) so extraction
    stays identical to a plain .html file."""

    def __init__(self, html_reader):
        self._html_reader = html_reader

    @staticmethod
    def _extract_html(file_path_str: str) -> Optional[str]:
        import email
        from email import policy

        with open(file_path_str, "rb") as handle:
            message = email.message_from_binary_file(handle, policy=policy.default)
        html_part = None
        text_part = None
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/html" and html_part is None:
                html_part = part
            elif content_type == "text/plain" and text_part is None:
                text_part = part
        chosen = html_part or text_part
        if chosen is None:
            return None
        content = chosen.get_content()
        return content if isinstance(content, str) else str(content)

    def load_data(self, file_path, **kwargs):
        file_path_str = str(file_path)
        file_name = os.path.basename(file_path_str)
        try:
            html = self._extract_html(file_path_str)
        except Exception as e:
            logger.error(f"MHTMLReader could not parse {file_name}: {e}")
            return []
        if not html:
            logger.info(f"MHTMLReader found no text/html or text/plain part in {file_name}.")
            return []
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".html", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(html)
                tmp_path = tmp.name
            docs = self._html_reader.load_data(tmp_path, **kwargs) or []
        except Exception as e:
            logger.error(f"MHTMLReader failed to extract HTML from {file_name}: {e}")
            return []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        # Rewrite provenance from the temp .html back to the original archive.
        for doc in docs:
            doc.metadata["file_path"] = file_path_str
            doc.metadata["file_name"] = file_name
        return docs
