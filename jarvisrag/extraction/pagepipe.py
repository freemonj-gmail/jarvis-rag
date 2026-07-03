"""Confidence-routed page extraction pipeline (OCR/scanned-doc enhancement, Phase 1).

This module is the shared, page-oriented backbone that the image reader
(``JpegExtractor``) and the scanned-PDF reader (``CombinedPDFReader``) delegate
to when ``JARVIS_ENABLE_PAGE_PIPELINE`` is set. Its job is to turn a single page
(a rasterized PDF page, or a standalone image) into an ordered list of
``Region`` objects, each carrying text, a provenance tag, and a confidence
score, so downstream chunking/embedding — and later phases — can reason about
*how* each span was extracted and *how much* to trust it.

Phase 1 stages (cheapest first; each escalates only the parts the prior stage
could not handle well):

1. **Native PDF text (PyMuPDF).** If a PDF page contains a real, extractable
   text layer, its blocks are emitted directly in natural reading order with
   full confidence. No OCR is run — this is both the fastest and the most
   accurate source.
2. **Surya layout + reading order + recognition.** For image-only / sparse
   pages, Surya provides typed layout regions in reading order and per-line
   recognition with confidence scores. (Optional; requires ``surya-ocr``.)
3. **Fallback OCR routing.** Only regions whose recognition confidence falls
   below ``JARVIS_OCR_MIN_CONF`` are re-OCR'd with the caller-supplied
   pytesseract/easyocr fallback, and the higher-confidence result wins.

Design constraints:
* The module imports cleanly even when PyMuPDF and/or Surya are absent — the
  optional backends are probed lazily and the pipeline degrades to whatever is
  available (ultimately to a single whole-page fallback-OCR call).
* Heavy models are loaded once and shared at class level, and only when the
  Surya stage is actually exercised, so importing this module has no runtime or
  memory cost.
* It has no dependency on the extractor/indexing layer (the readers pass their
  OCR fallback in as a callable), which keeps the import graph acyclic.

Later phases plug into the same ``Region`` stream: Phase 2 escalates
low-confidence / figure regions to a (non-Chinese) VLM such as Florence-2, and
Phase 3 runs a constrained local-LLM normalization pass over low-confidence
regions before embedding.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from jarvisrag.config import (
    JARVIS_ENABLE_SURYA,
    JARVIS_OCR_MIN_CONF,
    JARVIS_PDF_MIN_CHARS_PER_PAGE,
    logger,
)
from jarvisrag.text.textproc import normalize_text

# A fallback OCR callable takes a PIL image (a region crop or a whole page) and
# returns (text, confidence in 0.0-1.0). Confidence may be 0.0 when the engine
# does not expose one; the router still uses the text when Surya produced none.
FallbackOCR = Callable[["object"], Tuple[str, float]]


# ---------------------------------------------------------------------------
# Optional backend probes. Import failures are expected and non-fatal: the
# feature is opt-in and every stage has a graceful degradation path.
# ---------------------------------------------------------------------------
try:  # PyMuPDF — native PDF text + fast rasterization.
    import fitz  # type: ignore[import-not-found]

    pymupdf_available = True
except Exception:  # pragma: no cover - environment dependent
    fitz = None  # type: ignore[assignment]
    pymupdf_available = False


@dataclass
class Region:
    """A single extracted span of a page with provenance and confidence.

    ``bbox`` is ``(x0, y0, x1, y1)`` in pixel (image) or point (native PDF)
    coordinates. ``confidence`` is normalized to 0.0-1.0 (native text is 1.0).
    ``source`` records which engine produced the *current* text (e.g.
    ``"native_pymupdf"``, ``"surya"``, ``"ocr_fallback"``). ``reading_order`` is
    the 0-based position of the region in natural reading order.
    """

    text: str
    bbox: Tuple[float, float, float, float]
    region_type: str = "text"
    confidence: float = 0.0
    source: str = "unknown"
    reading_order: int = 0


@dataclass
class Page:
    """An ordered collection of regions extracted from one page/image."""

    regions: List[Region] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    page_number: Optional[int] = None
    # The dominant/native source for the page as a whole ("native_pymupdf",
    # "surya", "ocr_fallback"), useful for coarse Document metadata.
    source: str = "unknown"

    def ordered_regions(self) -> List[Region]:
        return sorted(self.regions, key=lambda r: r.reading_order)

    def text(self) -> str:
        """Assemble the page text in reading order."""
        parts = [r.text.strip() for r in self.ordered_regions() if r.text and r.text.strip()]
        return normalize_text("\n".join(parts))

    def mean_confidence(self) -> float:
        confs = [r.confidence for r in self.regions if r.text and r.text.strip()]
        return sum(confs) / len(confs) if confs else 0.0


# ---------------------------------------------------------------------------
# Native PDF text (PyMuPDF)
# ---------------------------------------------------------------------------
def page_native_char_count(fitz_page) -> int:
    """Number of extractable text characters on a PyMuPDF page (0 if none)."""
    try:
        return len((fitz_page.get_text("text") or "").strip())
    except Exception:
        return 0


def has_native_text_layer(fitz_page, min_chars: int = JARVIS_PDF_MIN_CHARS_PER_PAGE) -> bool:
    """True when the page carries a real, born-digital text layer.

    Uses a character-count floor rather than "any non-empty string" so that
    scanned PDFs with a few stray glyphs (page numbers, watermark text) are
    still correctly routed to OCR.
    """
    return page_native_char_count(fitz_page) >= max(int(min_chars), 1)


def native_text_page(fitz_page, page_number: Optional[int] = None) -> Optional[Page]:
    """Build a :class:`Page` from a PDF page's native text blocks in reading order.

    Returns ``None`` when the page has no usable text layer, signalling the
    caller to escalate to the OCR path.
    """
    if not has_native_text_layer(fitz_page):
        return None

    width = float(getattr(getattr(fitz_page, "rect", None), "width", 0.0) or 0.0)
    height = float(getattr(getattr(fitz_page, "rect", None), "height", 0.0) or 0.0)
    page = Page(width=width, height=height, page_number=page_number, source="native_pymupdf")

    try:
        data = fitz_page.get_text("dict")
        blocks = data.get("blocks", []) if isinstance(data, dict) else []
    except Exception:
        blocks = []

    collected: List[Tuple[Tuple[float, float], Region]] = []
    for block in blocks:
        if block.get("type", 0) != 0:  # 0 == text block; skip image blocks here
            continue
        lines_text: List[str] = []
        for line in block.get("lines", []):
            span_text = "".join(span.get("text", "") for span in line.get("spans", []))
            if span_text.strip():
                lines_text.append(span_text)
        block_text = "\n".join(lines_text).strip()
        if not block_text:
            continue
        x0, y0, x1, y1 = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        region = Region(
            text=block_text,
            bbox=(float(x0), float(y0), float(x1), float(y1)),
            region_type="text",
            confidence=1.0,
            source="native_pymupdf",
        )
        collected.append(((float(y0), float(x0)), region))

    if not collected:
        # Text existed at the char level but not as structured blocks; fall back
        # to a single whole-page region so we still capture it.
        raw = normalize_text(fitz_page.get_text("text") or "")
        if not raw:
            return None
        page.regions = [
            Region(text=raw, bbox=(0.0, 0.0, width, height), confidence=1.0, source="native_pymupdf")
        ]
        return page

    # Sort top-to-bottom, then left-to-right for natural reading order.
    collected.sort(key=lambda item: item[0])
    for order, (_, region) in enumerate(collected):
        region.reading_order = order
        page.regions.append(region)
    return page


# ---------------------------------------------------------------------------
# Surya layout + recognition
# ---------------------------------------------------------------------------
class SuryaLayoutEngine:
    """Lazy, process-wide Surya layout + recognition wrapper.

    The concrete Surya API has shifted across releases, so all model access is
    performed defensively (guarded imports, ``getattr`` field extraction). Any
    failure marks the engine unavailable and the caller falls back to plain OCR.
    Models are loaded once and shared at class level.
    """

    _initialized = False
    _available = False
    _detection = None
    _recognition = None
    _layout = None

    @classmethod
    def available(cls) -> bool:
        cls._ensure_loaded()
        return cls._available

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._initialized:
            return
        cls._initialized = True
        if not JARVIS_ENABLE_SURYA:
            cls._available = False
            return
        try:
            from surya.detection import DetectionPredictor  # type: ignore[import-not-found]
            from surya.recognition import RecognitionPredictor  # type: ignore[import-not-found]

            cls._detection = DetectionPredictor()
            cls._recognition = RecognitionPredictor()
            try:
                from surya.layout import LayoutPredictor  # type: ignore[import-not-found]

                cls._layout = LayoutPredictor()
            except Exception as e:  # layout is a nice-to-have; recognition can stand alone
                logger.info(f"Surya layout predictor unavailable ({e}); using recognition order only.")
                cls._layout = None
            cls._available = True
            logger.info("Surya layout/recognition initialized for page pipeline.")
        except Exception as e:
            logger.warning(f"Surya initialization failed; page pipeline will use OCR fallback only: {e}")
            cls._available = False

    @classmethod
    def _layout_regions(cls, image) -> List[Tuple[Tuple[float, float, float, float], str, int]]:
        """Return (bbox, label, reading_order) tuples from the layout predictor."""
        if cls._layout is None:
            return []
        try:
            results = cls._layout([image])
        except Exception as e:
            logger.info(f"Surya layout inference failed: {e}")
            return []
        if not results:
            return []
        out: List[Tuple[Tuple[float, float, float, float], str, int]] = []
        boxes = getattr(results[0], "bboxes", None) or []
        for i, box in enumerate(boxes):
            bbox = tuple(float(v) for v in (getattr(box, "bbox", None) or (0, 0, 0, 0)))
            label = str(getattr(box, "label", "text") or "text").lower()
            position = int(getattr(box, "position", i) or i)
            out.append((bbox, label, position))  # type: ignore[arg-type]
        return out

    @classmethod
    def _recognition_lines(cls, image) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
        """Return (text, bbox, confidence) for each recognized line, in engine order."""
        try:
            results = cls._recognition([image], det_predictor=cls._detection)
        except TypeError:
            # Older/newer positional signature: recognition(images, langs, det).
            try:
                results = cls._recognition([image], [None], cls._detection)
            except Exception as e:
                logger.info(f"Surya recognition inference failed: {e}")
                return []
        except Exception as e:
            logger.info(f"Surya recognition inference failed: {e}")
            return []
        if not results:
            return []
        lines_out: List[Tuple[str, Tuple[float, float, float, float], float]] = []
        text_lines = getattr(results[0], "text_lines", None) or []
        for line in text_lines:
            text = str(getattr(line, "text", "") or "")
            if not text.strip():
                continue
            bbox = tuple(float(v) for v in (getattr(line, "bbox", None) or (0, 0, 0, 0)))
            conf = getattr(line, "confidence", None)
            confidence = float(conf) if conf is not None else 0.0
            lines_out.append((text, bbox, confidence))  # type: ignore[arg-type]
        return lines_out

    @classmethod
    def extract(cls, image) -> List[Region]:
        """Run Surya on a PIL image, returning recognition regions in reading order.

        Layout, when available, supplies typed regions and a reading order; each
        recognized line is assigned to the layout region that contains its
        center. When layout is unavailable, the recognition engine's own
        top-to-bottom order is used.
        """
        cls._ensure_loaded()
        if not cls._available:
            return []
        lines = cls._recognition_lines(image)
        if not lines:
            return []
        layout = cls._layout_regions(image)

        regions: List[Region] = []
        if layout:
            # Group lines by containing layout box, preserving layout reading order.
            for text, bbox, confidence in lines:
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                best_order = None
                best_label = "text"
                for lbbox, label, position in layout:
                    if lbbox[0] <= cx <= lbbox[2] and lbbox[1] <= cy <= lbbox[3]:
                        best_order = position
                        best_label = label
                        break
                regions.append(
                    Region(
                        text=text,
                        bbox=bbox,
                        region_type=best_label,
                        confidence=confidence,
                        source="surya",
                        reading_order=best_order if best_order is not None else 10_000,
                    )
                )
            # Stabilize: order by (layout position, vertical, horizontal).
            regions.sort(key=lambda r: (r.reading_order, r.bbox[1], r.bbox[0]))
        else:
            for text, bbox, confidence in lines:
                regions.append(
                    Region(text=text, bbox=bbox, confidence=confidence, source="surya")
                )
            regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        for order, region in enumerate(regions):
            region.reading_order = order
        return regions


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _crop(image, bbox: Tuple[float, float, float, float]):
    """Crop a PIL image to bbox with a small margin; return None on failure."""
    try:
        w, h = image.size
        x0, y0, x1, y1 = bbox
        pad_x = max(2.0, (x1 - x0) * 0.02)
        pad_y = max(2.0, (y1 - y0) * 0.05)
        box = (
            max(0, int(x0 - pad_x)),
            max(0, int(y0 - pad_y)),
            min(w, int(x1 + pad_x)),
            min(h, int(y1 + pad_y)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        return image.crop(box)
    except Exception:
        return None


class PageProcessor:
    """Turns a page/image into a confidence-scored, reading-ordered :class:`Page`.

    ``fallback_ocr`` is a callable the caller supplies (wrapping its existing
    pytesseract/easyocr engines) used to re-OCR low-confidence regions and to
    handle the case where Surya is unavailable entirely.
    """

    def __init__(
        self,
        fallback_ocr: Optional[FallbackOCR] = None,
        min_confidence: float = JARVIS_OCR_MIN_CONF,
    ):
        self.fallback_ocr = fallback_ocr
        self.min_confidence = float(min_confidence)

    # -- PDF pages -------------------------------------------------------
    def process_pdf_page(self, fitz_page, page_number: Optional[int] = None) -> Page:
        """Prefer the native text layer; escalate image-only pages to OCR."""
        native = native_text_page(fitz_page, page_number=page_number)
        if native is not None:
            return native
        image = self._render_pdf_page(fitz_page)
        if image is None:
            return Page(page_number=page_number, source="unavailable")
        page = self.process_image(image)
        page.page_number = page_number
        return page

    def _render_pdf_page(self, fitz_page):
        from jarvisrag.config import JARVIS_PDF_OCR_DPI

        try:
            from PIL import Image

            pix = fitz_page.get_pixmap(dpi=int(JARVIS_PDF_OCR_DPI))
            return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        except Exception as e:
            logger.info(f"Failed to rasterize PDF page for OCR: {e}")
            return None

    # -- Images ----------------------------------------------------------
    def process_image(self, image) -> Page:
        """Run Surya (+ confidence-routed fallback) on a PIL image.

        When Surya is unavailable, the whole page is handed to the fallback OCR
        as a single region so the pipeline still produces output.
        """
        width, height = (0.0, 0.0)
        try:
            width, height = float(image.size[0]), float(image.size[1])
        except Exception:
            pass

        regions = SuryaLayoutEngine.extract(image) if JARVIS_ENABLE_SURYA else []

        if not regions:
            return self._whole_page_fallback(image, width, height)

        routed = self._route_low_confidence(image, regions)
        source = "surya"
        if any(r.source == "ocr_fallback" for r in routed):
            source = "surya+ocr_fallback"
        return Page(regions=routed, width=width, height=height, source=source)

    def _route_low_confidence(self, image, regions: List[Region]) -> List[Region]:
        if self.fallback_ocr is None:
            return regions
        for region in regions:
            if region.confidence >= self.min_confidence:
                continue
            crop = _crop(image, region.bbox)
            if crop is None:
                continue
            try:
                text, conf = self.fallback_ocr(crop)
            except Exception as e:
                logger.info(f"Fallback OCR failed on low-confidence region: {e}")
                continue
            text = (text or "").strip()
            if not text:
                continue
            # Keep whichever result we trust more; when the fallback engine gives
            # no confidence (0.0) but Surya was below threshold, prefer the
            # fallback text only if Surya produced nothing usable.
            if conf >= region.confidence or not region.text.strip():
                region.text = text
                region.confidence = max(conf, region.confidence)
                region.source = "ocr_fallback"
        return regions

    def _whole_page_fallback(self, image, width: float, height: float) -> Page:
        if self.fallback_ocr is None:
            return Page(width=width, height=height, source="unavailable")
        try:
            text, conf = self.fallback_ocr(image)
        except Exception as e:
            logger.info(f"Whole-page fallback OCR failed: {e}")
            return Page(width=width, height=height, source="unavailable")
        text = (text or "").strip()
        if not text:
            return Page(width=width, height=height, source="unavailable")
        region = Region(
            text=text,
            bbox=(0.0, 0.0, width, height),
            confidence=conf,
            source="ocr_fallback",
        )
        return Page(regions=[region], width=width, height=height, source="ocr_fallback")


def open_pdf(file_path: str):
    """Open a PDF with PyMuPDF, or return ``None`` when unavailable."""
    if not pymupdf_available:
        return None
    try:
        return fitz.open(file_path)
    except Exception as e:
        logger.info(f"PyMuPDF could not open {file_path}: {e}")
        return None


def region_metadata(region: Region) -> dict:
    """Provenance/confidence metadata for a region, merged into Document metadata."""
    return {
        "extracted_by": region.source,
        "region_type": region.region_type,
        "ocr_confidence": round(float(region.confidence), 4),
        "reading_order_index": region.reading_order,
    }
