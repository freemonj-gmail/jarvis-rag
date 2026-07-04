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
    """Order recognized lines into natural reading order and join them.

    ``ocrmac`` returns ``(text, confidence, [x, y, w, h])`` per line, where the
    bounding box is normalized to 0.0-1.0 with a *bottom-left* origin (Vision
    convention): a larger ``y`` is higher on the page. We therefore sort by
    descending ``y`` (top-to-bottom) then ascending ``x`` (left-to-right).
    """
    items: List[Tuple[float, float, str]] = []
    confidences: List[float] = []
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
        except Exception:
            continue
        if not text:
            continue
        items.append((y, x, text))
        confidences.append(confidence)

    if not items:
        return "", 0.0

    items.sort(key=lambda item: (-item[0], item[1]))
    text = "\n".join(item[2] for item in items)
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, confidence


def apple_ocr_available() -> bool:
    """True when Apple Vision OCR is enabled, on macOS, and importable."""
    return AppleOCREngine._ensure_available()


def apple_ocr(image) -> Tuple[str, float]:
    """Verbatim-transcribe a PIL image with Apple Vision; ``(text, confidence)``."""
    return AppleOCREngine.recognize(image)
