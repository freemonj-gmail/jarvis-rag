"""PDF readers: native text, tables, and the OCR fallback for scanned pages.

Also turns a page-pipeline ``Page`` into Documents (shared with image OCR).
"""

import os
import shutil
import subprocess
import time
from typing import List

import easyocr
import numpy as np
import pandas as pd
from llama_index.core.schema import Document

from jarvisrag.config import JARVIS_ENABLE_PAGE_PIPELINE, logger
from jarvisrag.extraction.indexing import chunk_text
from jarvisrag.extraction.pagepipe import open_pdf, Page, PageProcessor, pymupdf_available
from jarvisrag.iotools import suppress_java_warnings, time_limit
from jarvisrag.text.textproc import normalize_text


def _documents_from_page(page: "Page", file_path_str: str, file_name: str) -> List["Document"]:
    """Chunk a page-pipeline :class:`Page` into Documents in reading order.

    Page-level provenance (which engine produced the text) and the mean
    recognition confidence are recorded in metadata so later phases can prefer
    high-confidence native text over rescued OCR when ranking.
    """
    text = page.text()
    if not text:
        return []
    confidence = round(page.mean_confidence(), 4)
    docs: List[Document] = []
    for chunk_index, chunk in enumerate(chunk_text(text)):
        metadata = {
            "file_path": file_path_str,
            "file_name": file_name,
            "chunk_index": chunk_index,
            "extracted_by": page.source,
            "ocr_confidence": confidence,
        }
        if page.page_number is not None:
            metadata["page_number"] = page.page_number
        docs.append(Document(text=chunk, metadata=metadata))
    return docs


class TablePDFReader:
    def __init__(self):
        self._tabula_available = None
        self._tabula = None

    def _java_available(self) -> bool:
        if shutil.which("java") is None:
            return False
        try:
            result = subprocess.run(
                ["java", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_tabula_available(self):
        if self._tabula_available is not None:
            return self._tabula_available

        if not self._java_available():
            logger.info("Java/JDK is not installed; disabling tabula-py table extraction.")
            self._tabula_available = False
            return False

        try:
            import tabula
            self._tabula = tabula
            self._tabula_available = True
            return True
        except ImportError:
            logger.info("tabula-py is not installed; disabling tabula-py table extraction.")
            self._tabula_available = False
            return False
        except Exception as e:
            logger.info(f"tabula-py initialization failed: {e}. Disabling tabula-py table extraction.")
            self._tabula_available = False
            return False

    def _disable_tabula(self, reason: str):
        if self._tabula_available is not False:
            logger.info(f"Disabling tabula-py table extraction: {reason}")
        self._tabula_available = False
        self._tabula = None

    def load_data(self, file_path: str, **kwargs):
        docs = []
        file_name = os.path.basename(file_path)

        if not self._check_tabula_available():
            tabula = None
        else:
            tabula = self._tabula

        if tabula is not None:
            try:
                with suppress_java_warnings():
                    tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
                for i, table in enumerate(tables):
                    if not table.empty:
                        table_text = table.to_string(index=False)
                        if table_text.strip():
                            docs.append(Document(text=table_text, metadata={
                                "file_path": file_path,
                                "file_name": file_name,
                                "page_number": None,
                                "table_index": i,
                                "extracted_by": "tabula"
                            }))
                if docs:
                    return docs
            except Exception as e:
                err = str(e).lower()
                if any(keyword in err for keyword in ["jvm dll not found", "jvm", "java", "jpype"]):
                    self._disable_tabula(
                        "Java/JVM is unavailable or incorrectly configured for tabula-py. "
                        "Install a valid JDK only if you need tabula table extraction."
                    )
                else:
                    logger.info(f"Note: tabula table extraction failed for {file_path}: {e}")

        try:
            with suppress_java_warnings():
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        page_tables = page.extract_tables()
                        for i, table in enumerate(page_tables):
                            if table:
                                table_text = pd.DataFrame(table).to_string(index=False)
                                if table_text.strip():
                                    docs.append(Document(text=table_text, metadata={
                                        "file_path": file_path,
                                        "file_name": file_name,
                                        "page_number": page_num,
                                        "table_index": i,
                                        "extracted_by": "pdfplumber_table"
                                    }))
        except Exception as e:
            logger.info(f"Note: pdfplumber table extraction failed for {file_path}: {e}")
        return docs


class CombinedPDFReader:
    def __init__(self):
        self.table_reader = TablePDFReader()
        self.easyocr_reader = None
        self.pdf_file_timeout_seconds = float(os.environ.get("JARVIS_PDF_FILE_TIMEOUT_SECONDS", "90"))

    def _get_easyocr_reader(self):
        if self.easyocr_reader is None:
            logger.info("Initializing PDF EasyOCR fallback model...")
            self.easyocr_reader = easyocr.Reader(['en'])
            logger.info("PDF EasyOCR fallback model initialized.")
        return self.easyocr_reader

    def _build_page_docs(self, text: str, metadata: dict, source: str):
        docs = []
        chunks = chunk_text(text)
        for chunk_index, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_index": chunk_index,
                "source": source,
            })
            docs.append(Document(text=chunk, metadata=chunk_meta))
        return docs

    def _pipeline_fallback_ocr(self, crop):
        """easyocr-based fallback for the page pipeline; returns (text, conf)."""
        try:
            reader = self._get_easyocr_reader()
            result = reader.readtext(np.array(crop), detail=1)
            texts = [r[1] for r in result if len(r) >= 2 and r[1]]
            confs = [float(r[2]) for r in result if len(r) >= 3]
            text = normalize_text("\n".join(texts))
            if text:
                return text, (sum(confs) / len(confs) if confs else 0.0)
        except Exception as e:
            logger.info(f"PDF pipeline easyocr fallback failed: {e}")
        return "", 0.0

    def _load_with_pipeline(self, file_path: str, file_name: str, **kwargs):
        """Confidence-routed path: native PyMuPDF text per page, OCR only for
        image-only/sparse pages. Returns ``None`` if the PDF can't be opened
        (signalling the caller to use the legacy pdfplumber path)."""
        doc = open_pdf(file_path)
        if doc is None:
            return None
        all_docs: List[Document] = []
        processor = PageProcessor(fallback_ocr=self._pipeline_fallback_ocr)
        try:
            with time_limit(self.pdf_file_timeout_seconds, f"PDF pipeline for {file_name}"):
                for page_index in range(doc.page_count):
                    fitz_page = doc.load_page(page_index)
                    page = processor.process_pdf_page(fitz_page, page_number=page_index + 1)
                    all_docs.extend(_documents_from_page(page, file_path, file_name))
        except TimeoutError as e:
            logger.warning(f"{e}. Returning partial content for {file_name}.")
        except Exception as e:
            logger.info(f"PDF pipeline extraction error for {file_name}: {e}")
        finally:
            try:
                doc.close()
            except Exception:
                pass
        # Tables stay on pdfplumber/tabula — they extract tabular structure far
        # better than page OCR does.
        try:
            table_docs = self.table_reader.load_data(file_path, **kwargs)
            if table_docs:
                all_docs.extend(table_docs)
        except Exception as e:
            logger.info(f"Table extraction failed for {file_name}: {e}")
        return all_docs

    def load_data(self, file_path: str, **kwargs):
        import pdfplumber
        all_docs = []
        file_name = os.path.basename(file_path)
        started = time.time()
        logger.info(f"Reading PDF: {file_name}")

        # -----------------------------------------
        # CONFIDENCE-ROUTED PAGE PIPELINE (opt-in)
        # -----------------------------------------
        if JARVIS_ENABLE_PAGE_PIPELINE and pymupdf_available:
            try:
                pipeline_docs = self._load_with_pipeline(file_path, file_name, **kwargs)
                if pipeline_docs:
                    logger.info(
                        f"Finished PDF (pipeline): {file_name} -> "
                        f"{len(pipeline_docs)} chunks in {time.time() - started:.1f}s"
                    )
                    return pipeline_docs
                if pipeline_docs is not None:
                    logger.info(
                        f"Page pipeline produced no content for {file_name}; "
                        f"falling back to the legacy PDF path."
                    )
            except Exception as e:
                logger.warning(
                    f"Page pipeline failed for {file_name} ({e}); "
                    f"falling back to the legacy PDF path."
                )

        try:
            with time_limit(self.pdf_file_timeout_seconds, f"PDF extraction for {file_name}"):
                with suppress_java_warnings():
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages, start=1):
                            page_text = normalize_text(page.extract_text())
                            page_metadata = {
                                "file_path": file_path,
                                "file_name": file_name,
                                "page_number": page_num,
                            }
                            if page_text:
                                all_docs.extend(self._build_page_docs(page_text, {**page_metadata, "extracted_by": "pdfplumber"}, "pdfplumber"))
                    table_docs = self.table_reader.load_data(file_path, **kwargs)
                    if table_docs:
                        all_docs.extend(table_docs)
        except TimeoutError as e:
            logger.warning(f"{e}. Returning partial content for {file_name}.")
        except Exception as e:
            logger.info(f"pdfplumber extraction error: {e}")

        if not all_docs:
            logger.info(f"No text extracted from {file_name} via text/table extraction. Attempting OCR fallback...")
            try:
                with time_limit(self.pdf_file_timeout_seconds, f"OCR extraction for {file_name}"):
                    with suppress_java_warnings():
                        with pdfplumber.open(file_path) as pdf:
                            for page_num, page in enumerate(pdf.pages, start=1):
                                page_metadata = {
                                    "file_path": file_path,
                                    "file_name": file_name,
                                    "page_number": page_num,
                                }
                                try:
                                    page_image = page.to_image(resolution=300).original.convert("RGB")
                                    image_array = np.array(page_image)
                                    result = self._get_easyocr_reader().readtext(image_array, detail=0)
                                    page_text = "\n".join(result)
                                    if page_text.strip():
                                        all_docs.extend(self._build_page_docs(page_text, {**page_metadata, "extracted_by": "easyocr"}, "easyocr"))
                                except Exception as e:
                                    logger.info(f"EasyOCR failed on page {page_num} of {file_path}: {e}")
            except TimeoutError as e:
                logger.warning(f"{e}. Returning partial OCR content for {file_name}.")
            except Exception as e:
                logger.info(f"OCR extraction error: {e}")

        if not all_docs:
            logger.warning(f"No content extracted from {file_path}")
        logger.info(
            f"Finished PDF: {file_name} -> {len(all_docs)} chunks in {time.time() - started:.1f}s"
        )
        return all_docs
