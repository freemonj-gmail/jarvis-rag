"""Apple Vision OCR backend for the page pipeline (macOS on-device text recognition).

Uses Apple's Vision framework (via the ``ocrmac`` wrapper around
``VNRecognizeTextRequest``) to transcribe text from images *verbatim*. Unlike a
general vision-language model, Vision OCR does not paraphrase or fabricate field
values, and it is hardware-accelerated (sub-second per image, no timeouts),
which makes it the preferred text engine for scanned/photographed documents such
as IDs, receipts, and forms. The VLM is then reserved for *describing* visual
content (see ``JARVIS_VLM_CAPTION``).

macOS-only and opt-in (``JARVIS_ENABLE_APPLE_OCR``). Guarded and lazily imported
like the other optional backends: the availability probe is cached, and any
platform mismatch or import/inference failure marks the engine unavailable so
the caller falls back to its existing OCR/VLM path. This module never raises.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from jarvisrag.config import (
    JARVIS_APPLE_OCR_LANGS,
    JARVIS_APPLE_OCR_LEVEL,
    JARVIS_ENABLE_APPLE_OCR,
    logger,
)


def _language_preference() -> List[str]:
    langs = [part.strip() for part in JARVIS_APPLE_OCR_LANGS.split(",") if part.strip()]
    return langs or ["en-US"]


def _recognition_level() -> str:
    level = (JARVIS_APPLE_OCR_LEVEL or "").strip().lower()
    return level if level in ("accurate", "fast") else "accurate"


class AppleOCREngine:
    """Lazy, guarded wrapper around Apple's Vision text recognizer.

    The ``ocrmac`` import (and the underlying pyobjc Vision framework) is probed
    only when the feature is enabled and only on macOS; the result is cached on
    the class so repeated calls are cheap and importing this module has no cost.
    """

    _available: Optional[bool] = None

    @classmethod
    def _ensure_available(cls) -> bool:
        if cls._available is not None:
            return cls._available
        if not JARVIS_ENABLE_APPLE_OCR:
            cls._available = False
            return False
        if sys.platform != "darwin":
            logger.info("Apple Vision OCR requested but not running on macOS; disabling.")
            cls._available = False
            return False
        try:
            from ocrmac import ocrmac  # noqa: F401  (probe import only)

            cls._available = True
            logger.info("Apple Vision OCR backend available.")
        except Exception as e:  # pragma: no cover - environment dependent
            logger.info(f"Apple Vision OCR unavailable ({e}); disabling.")
            cls._available = False
        return cls._available

    @classmethod
    def recognize(cls, image) -> Tuple[str, float]:
        """Transcribe ``image`` verbatim; return ``(text, mean_confidence)``.

        Returns ``("", 0.0)`` when the engine is unavailable, the image is empty,
        or recognition fails/yields nothing.
        """
        if image is None or not cls._ensure_available():
            return "", 0.0
        try:
            from ocrmac import ocrmac
            from PIL import ImageOps

            rgb = image.convert("RGB") if getattr(image, "mode", None) != "RGB" else image
            # Phone photos carry an EXIF orientation tag; bake it in so Vision sees
            # upright pixels (it recognizes from the raw buffer, not the tag).
            rgb = ImageOps.exif_transpose(rgb)
            annotations = ocrmac.OCR(
                rgb,
                recognition_level=_recognition_level(),
                language_preference=_language_preference(),
            ).recognize()
        except Exception as e:  # pragma: no cover - environment dependent
            logger.info(f"Apple Vision OCR failed: {e}")
            return "", 0.0
        return _assemble(annotations)


def _assemble(annotations) -> Tuple[str, float]:
    """Reconstruct reading order from Vision's line boxes, preserving rows.

    ``ocrmac`` returns ``(text, confidence, [x, y, w, h])`` per recognized line,
    normalized 0.0-1.0 with a *bottom-left* origin (Vision convention): a larger
    ``y`` is higher on the page.

    A naive global sort by ``y`` interleaves cells from different columns of a
    table (e.g. a scanned transcript) into one flat stream, destroying the
    row/column association. Instead we group lines into horizontal *rows* (band
    the page by vertical position, using the line height as the tolerance), then
    order each row left-to-right by ``x``. Cells that share a visual row stay
    together on one output line, and rows stay in top-to-bottom order — so
    tabular data (a label next to its value) remains readable for downstream
    retrieval and synthesis, while ordinary prose is unaffected (each paragraph
    line is simply its own single-cell row).
    """
    # (center_y, x, height, text) per line.
    items: List[Tuple[float, float, float, str]] = []
    confidences: List[float] = []
    heights: List[float] = []
    for annotation in annotations or []:
        try:
            text = str(annotation[0] or "").strip()
            confidence = (
                float(annotation[1])
                if len(annotation) > 1 and annotation[1] is not None
                else 0.0
            )
            bbox = annotation[2] if len(annotation) > 2 else (0.0, 0.0, 0.0, 0.0)
            x = float(bbox[0])
            y = float(bbox[1])
            h = float(bbox[3]) if len(bbox) > 3 else 0.0
        except Exception:
            continue
        if not text:
            continue
        items.append((y + h / 2.0, x, h, text))
        confidences.append(confidence)
        if h > 0:
            heights.append(h)

    if not items:
        return "", 0.0

    # Row-band tolerance: half the median line height (fall back to a small
    # constant when Vision does not report usable heights). Two lines whose
    # vertical centers are within this distance are treated as the same row.
    if heights:
        heights.sort()
        median_h = heights[len(heights) // 2]
    else:
        median_h = 0.0
    tol = (median_h / 2.0) if median_h > 0 else 0.012

    # Sort top-to-bottom (descending center_y), then cluster into rows.
    items.sort(key=lambda item: -item[0])
    rows: List[List[Tuple[float, float, float, str]]] = []
    current: List[Tuple[float, float, float, str]] = []
    band_center: Optional[float] = None
    for item in items:
        center_y = item[0]
        if band_center is None or abs(center_y - band_center) <= tol:
            current.append(item)
            # Running mean keeps the band anchored as lines accrete.
            band_center = (
                center_y
                if band_center is None
                else (band_center * (len(current) - 1) + center_y) / len(current)
            )
        else:
            rows.append(current)
            current = [item]
            band_center = center_y
    if current:
        rows.append(current)

    lines: List[str] = []
    for row in rows:
        row.sort(key=lambda item: item[1])  # left-to-right within the row
        lines.append(" ".join(item[3] for item in row))
    text = "\n".join(lines)
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, confidence


def apple_ocr_available() -> bool:
    """True when Apple Vision OCR is enabled, on macOS, and importable."""
    return AppleOCREngine._ensure_available()


def apple_ocr(image) -> Tuple[str, float]:
    """Verbatim-transcribe a PIL image with Apple Vision; ``(text, confidence)``."""
    return AppleOCREngine.recognize(image)
