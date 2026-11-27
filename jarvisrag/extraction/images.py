"""Image text extraction: OCR engines and the optional vision-language model.

Surya and TrOCR are optional; each is used only when its package imports.
"""

import os
import re
from typing import List

import cv2
import easyocr
import numpy as np
import pytesseract
from PIL import Image
from llama_index.core.schema import Document

from jarvisrag.config import JARVIS_ENABLE_PAGE_PIPELINE, logger
from jarvisrag.extraction.indexing import chunk_text
from jarvisrag.extraction.pagepipe import PageProcessor
from jarvisrag.text.textproc import normalize_text
from jarvisrag.extraction.pdf import _documents_from_page


# Surya / TrOCR optional imports
try:
    from surya.ocr import run_ocr  # type: ignore[import-not-found]
    from surya.model.detection import load_model as load_detection_model  # type: ignore[import-not-found]
    from surya.model.recognition import load_model as load_recognition_model  # type: ignore[import-not-found]
    surya_available = True
except Exception:
    surya_available = False


try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    trocr_available = True
except Exception:
    trocr_available = False


class JpegExtractor:

    # -------------------------------------------------
    # CLASS-LEVEL SHARED MODELS
    # -------------------------------------------------

    _easyocr_reader = None

    _surya_det_model = None
    _surya_rec_model = None

    _trocr_processor = None
    _trocr_model = None

    _models_initialized = False

    # -------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------

    def __init__(self):

        if not JpegExtractor._models_initialized:

            logger.info("Initializing OCR models (one-time startup)...")

            # -----------------------------------------
            # EasyOCR
            # -----------------------------------------
            try:
                JpegExtractor._easyocr_reader = easyocr.Reader(
                    ['en'],
                    gpu=False
                )

                logger.info("EasyOCR initialized.")

            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")

            # -----------------------------------------
            # Surya
            # -----------------------------------------
            if surya_available:

                try:
                    JpegExtractor._surya_det_model = load_detection_model()

                    JpegExtractor._surya_rec_model = (
                        load_recognition_model()
                    )

                    logger.info("Surya OCR initialized.")

                except Exception as e:
                    logger.warning(
                        f"Surya initialization failed: {e}"
                    )

            # -----------------------------------------
            # TrOCR
            # -----------------------------------------
            if trocr_available:

                try:

                    logger.info(
                        "Loading TrOCR processor..."
                    )

                    JpegExtractor._trocr_processor = (
                        TrOCRProcessor.from_pretrained(
                            'microsoft/trocr-base-printed'
                        )
                    )

                    logger.info(
                        "Loading TrOCR model..."
                    )

                    JpegExtractor._trocr_model = (
                        VisionEncoderDecoderModel.from_pretrained(
                            'microsoft/trocr-base-printed'
                        )
                    )

                    logger.info("TrOCR initialized.")

                except Exception as e:
                    logger.warning(
                        f"TrOCR initialization failed: {e}"
                    )

            JpegExtractor._models_initialized = True

    # -------------------------------------------------
    # IMAGE PREPROCESSING
    # -------------------------------------------------

    def _preprocess_image(self, image):

        cv_image = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

        gray = cv2.cvtColor(
            cv_image,
            cv2.COLOR_BGR2GRAY
        )

        h, w = gray.shape[:2]

        max_dim = max(h, w)

        if max_dim < 1600:

            scale = 1600 / max_dim

            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        denoised = cv2.fastNlMeansDenoising(
            enhanced,
            h=3
        )

        blurred = cv2.medianBlur(
            denoised,
            3
        )

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        return Image.fromarray(thresh)

    # -------------------------------------------------
    # IMAGE LOAD
    # -------------------------------------------------

    def _load_image(self, file_path_str):

        try:

            image = Image.open(file_path_str)

            return image.convert("RGB")

        except Exception as e:

            logger.error(
                f"Unable to open image {file_path_str}: {e}"
            )

            return None

    # -------------------------------------------------
    # SURYA OCR
    # -------------------------------------------------

    def _ocr_with_surya(self, image):

        if (
            not surya_available
            or JpegExtractor._surya_det_model is None
            or JpegExtractor._surya_rec_model is None
        ):
            return None

        try:

            predictions = run_ocr(
                [image],
                [JpegExtractor._surya_det_model],
                [JpegExtractor._surya_rec_model],
                langs=['en']
            )

            if predictions and predictions[0]:

                text = "\n".join(
                    [line.text for line in predictions[0]]
                )

                return text

        except Exception as e:

            logger.error(f"Surya OCR failed: {e}")

        return None

    # -------------------------------------------------
    # TrOCR
    # -------------------------------------------------

    def _ocr_with_trocr(self, image):

        if (
            not trocr_available
            or JpegExtractor._trocr_processor is None
            or JpegExtractor._trocr_model is None
        ):
            return None

        try:

            pixel_values = (
                JpegExtractor._trocr_processor(
                    image,
                    return_tensors="pt"
                ).pixel_values
            )

            generated_ids = (
                JpegExtractor._trocr_model.generate(
                    pixel_values,
                    max_new_tokens=256
                )
            )

            generated_text = (
                JpegExtractor._trocr_processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
            )

            return generated_text

        except Exception as e:

            logger.error(f"TrOCR failed: {e}")

        return None

    # -------------------------------------------------
    # OCR QUALITY CHECK
    # -------------------------------------------------

    def _is_valid_ocr_text(self, text: str) -> bool:

        if not text:
            return False

        normalized = normalize_text(text)

        if len(normalized) < 20:

            keywords = [
                "dob",
                "date",
                "birth",
                "license",
                "name",
                "expiry",
                "issue",
                "address",
            ]

            lower = normalized.lower()

            if (
                any(keyword in lower for keyword in keywords)
                and len(re.findall(r"\w+", normalized)) >= 2
            ):
                return True

            return False

        if re.fullmatch(r"[\W_]+", normalized):
            return False

        tokens = re.findall(r"\w+", normalized)

        return len(tokens) >= 3

    # -------------------------------------------------
    # PAGE-PIPELINE FALLBACK OCR
    # -------------------------------------------------

    def _pipeline_fallback_ocr(self, crop):
        """Re-OCR a low-confidence region crop for the page pipeline.

        Tries pytesseract (fast, exposes per-word confidence) then easyocr, and
        returns ``(text, confidence)`` with confidence normalized to 0.0-1.0.
        """
        try:
            processed = self._preprocess_image(crop)
        except Exception:
            processed = crop

        try:
            data = pytesseract.image_to_data(
                processed, output_type=pytesseract.Output.DICT
            )
            words: List[str] = []
            confs: List[float] = []
            for word, conf in zip(data.get("text", []), data.get("conf", [])):
                if word and word.strip():
                    words.append(word)
                    try:
                        c = float(conf)
                    except (TypeError, ValueError):
                        c = -1.0
                    if c >= 0:
                        confs.append(c / 100.0)
            text = normalize_text(" ".join(words))
            if text:
                return text, (sum(confs) / len(confs) if confs else 0.0)
        except Exception as e:
            logger.info(f"pytesseract fallback failed: {e}")

        try:
            reader = JpegExtractor._easyocr_reader
            if reader is not None:
                result = reader.readtext(np.array(processed), detail=1)
                texts = [r[1] for r in result if len(r) >= 2 and r[1]]
                confs = [float(r[2]) for r in result if len(r) >= 3]
                text = normalize_text("\n".join(texts))
                if text:
                    return text, (sum(confs) / len(confs) if confs else 0.0)
        except Exception as e:
            logger.info(f"easyocr fallback failed: {e}")

        return "", 0.0

    # -------------------------------------------------
    # MAIN LOAD
    # -------------------------------------------------

    def load_data(self, file_path, **kwargs):

        file_path_str = str(file_path)

        file_name = os.path.basename(file_path)

        documents = []

        image = self._load_image(file_path_str)

        if image is None:
            return documents

        # -----------------------------------------
        # CONFIDENCE-ROUTED PAGE PIPELINE (opt-in)
        # -----------------------------------------
        # When enabled, run Surya layout + reading order + per-region
        # recognition confidence, routing only the low-confidence regions to the
        # pytesseract/easyocr fallback below. Falls back to the legacy cascade if
        # the pipeline yields nothing or errors.
        if JARVIS_ENABLE_PAGE_PIPELINE:
            try:
                processor = PageProcessor(fallback_ocr=self._pipeline_fallback_ocr)
                page = processor.process_image(image)
                pipeline_docs = _documents_from_page(page, file_path_str, file_name)
                if pipeline_docs:
                    return pipeline_docs
                logger.info(
                    f"Page pipeline produced no text for {file_name}; "
                    f"falling back to the legacy OCR cascade."
                )
            except Exception as e:
                logger.warning(
                    f"Page pipeline failed for {file_name} ({e}); "
                    f"falling back to the legacy OCR cascade."
                )

        processed_image = self._preprocess_image(image)

        # -----------------------------------------
        # SURYA
        # -----------------------------------------

        text_surya = self._ocr_with_surya(
            processed_image
        )

        if text_surya and text_surya.strip():

            text = normalize_text(text_surya)

            for chunk_index, chunk in enumerate(
                chunk_text(text)
            ):

                documents.append(Document(
                    text=chunk,
                    metadata={
                        "file_path": file_path_str,
                        "file_name": file_name,
                        "chunk_index": chunk_index,
                        "extracted_by": "surya"
                    }
                ))

            return documents

        # -----------------------------------------
        # TrOCR
        # -----------------------------------------

        text_trocr = self._ocr_with_trocr(
            processed_image
        )

        if text_trocr and text_trocr.strip():

            if self._is_valid_ocr_text(text_trocr):

                text = normalize_text(text_trocr)

                for chunk_index, chunk in enumerate(
                    chunk_text(text)
                ):

                    documents.append(Document(
                        text=chunk,
                        metadata={
                            "file_path": file_path_str,
                            "file_name": file_name,
                            "chunk_index": chunk_index,
                            "extracted_by": "trocr"
                        }
                    ))

                return documents

            logger.info(
                f"TrOCR result for {file_name} "
                f"appears too short/low-quality; "
                f"falling back to additional OCR engines. "
                f"Output: {repr(text_trocr)}"
            )

        # -----------------------------------------
        # PYTESSERACT
        # -----------------------------------------

        pyt_docs = []

        try:

            text_pt = normalize_text(
                pytesseract.image_to_string(
                    processed_image
                )
            )

            if text_pt:

                for chunk_index, chunk in enumerate(
                    chunk_text(text_pt)
                ):

                    pyt_docs.append(Document(
                        text=chunk,
                        metadata={
                            "file_path": file_path_str,
                            "file_name": file_name,
                            "chunk_index": chunk_index,
                            "extracted_by": "pytesseract"
                        }
                    ))

        except Exception as e:

            logger.error(
                f"pytesseract failed on {file_path}: {e}"
            )

        # -----------------------------------------
        # EASYOCR
        # -----------------------------------------

        eo_docs = []

        try:

            image_array = np.array(processed_image)

            result = (
                JpegExtractor._easyocr_reader.readtext(
                    image_array,
                    detail=0
                )
            )

            text_eo = normalize_text(
                "\n".join(result)
            )

            if text_eo:

                for chunk_index, chunk in enumerate(
                    chunk_text(text_eo)
                ):

                    eo_docs.append(Document(
                        text=chunk,
                        metadata={
                            "file_path": file_path_str,
                            "file_name": file_name,
                            "chunk_index": chunk_index,
                            "extracted_by": "easyocr"
                        }
                    ))

        except Exception as e:

            logger.error(
                f"easyocr failed on {file_path}: {e}"
            )

        documents.extend(pyt_docs)
        documents.extend(eo_docs)

        return documents
