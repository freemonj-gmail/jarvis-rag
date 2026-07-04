"""Vision-language model (VLM) escalation for the page pipeline (OCR Phase 2).

Wraps a local, non-Chinese vision-language model — Florence-2 (Microsoft, MIT)
by default — behind a small, defensive interface used by ``PageProcessor`` to:

* re-OCR regions the cheaper stages (Surya recognition / pytesseract / easyocr)
  could not read confidently, and
* optionally caption figure/photo regions so non-text visual content becomes
  searchable.

Design mirrors the other optional backends in this package:

* Nothing heavy is imported at module load. ``torch``/``transformers`` are
  imported lazily the first time the engine is actually used, and only when
  ``JARVIS_ENABLE_VLM`` is set.
* The model is loaded once and shared at class level.
* Every load/inference path is guarded; any failure marks the engine
  unavailable and the caller falls back to its OCR result. A VLM problem must
  never break extraction.

Apple Silicon notes: Florence-2's reference code opportunistically imports
``flash_attn`` (which has no MPS wheel); we strip that import at load time and
run with the eager attention path in float32 on MPS/CPU.

The interface is model-agnostic on purpose: swap ``JARVIS_VLM_MODEL`` to another
non-Chinese VLM (PaliGemma 2, Kosmos-2.5, Llama 3.2 Vision, ...) without touching
the pipeline. A stronger full-page model becomes worthwhile on higher-memory
Apple Silicon (M-series Pro/Max/Ultra).
"""

from __future__ import annotations

from typing import Optional

from jarvisrag.config import (
    JARVIS_ENABLE_VLM,
    JARVIS_VLM_MODEL,
    logger,
)

# Florence-2 task prompts.
_OCR_TASK = "<OCR>"
_CAPTION_TASK = "<MORE_DETAILED_CAPTION>"


def _patched_get_imports():
    """Return a context manager that strips ``flash_attn`` from a remote module's
    import probe, so Florence-2 loads on machines without a flash-attn wheel
    (e.g. Apple Silicon). No-op-safe if the internals are unavailable."""
    from contextlib import contextmanager

    @contextmanager
    def _noop():
        yield

    try:
        from unittest.mock import patch
        from transformers.dynamic_module_utils import get_imports  # type: ignore[import-not-found]
    except Exception:
        return _noop()

    def _fixed(filename):
        try:
            imports = list(get_imports(filename))
        except Exception:
            return get_imports(filename)
        return [imp for imp in imports if imp != "flash_attn"]

    return patch("transformers.dynamic_module_utils.get_imports", _fixed)


class VLMEngine:
    """Lazy, process-wide Florence-2 (or compatible) VLM wrapper."""

    _initialized = False
    _available = False
    _model = None
    _processor = None
    _device = "cpu"
    _dtype = None

    @classmethod
    def available(cls) -> bool:
        cls._ensure_loaded()
        return cls._available

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._initialized:
            return
        cls._initialized = True
        if not JARVIS_ENABLE_VLM:
            cls._available = False
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore[import-not-found]

            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                cls._device, cls._dtype = "mps", torch.float32
            elif torch.cuda.is_available():
                cls._device, cls._dtype = "cuda", torch.float16
            else:
                cls._device, cls._dtype = "cpu", torch.float32

            logger.info(f"Loading VLM '{JARVIS_VLM_MODEL}' on {cls._device} (one-time)...")
            with _patched_get_imports():
                cls._model = AutoModelForCausalLM.from_pretrained(
                    JARVIS_VLM_MODEL,
                    trust_remote_code=True,
                    torch_dtype=cls._dtype,
                )
                cls._processor = AutoProcessor.from_pretrained(
                    JARVIS_VLM_MODEL,
                    trust_remote_code=True,
                )
            cls._model = cls._model.to(cls._device).eval()
            cls._available = True
            logger.info("VLM initialized for page pipeline.")
        except Exception as e:
            logger.warning(f"VLM initialization failed; page pipeline will skip VLM escalation: {e}")
            cls._available = False

    @classmethod
    def _run(cls, image, task_prompt: str) -> str:
        cls._ensure_loaded()
        if not cls._available:
            return ""
        try:
            import torch  # type: ignore[import-not-found]

            rgb = image.convert("RGB") if getattr(image, "mode", "RGB") != "RGB" else image
            inputs = cls._processor(text=task_prompt, images=rgb, return_tensors="pt")
            input_ids = inputs["input_ids"].to(cls._device)
            pixel_values = inputs["pixel_values"].to(cls._device, cls._dtype)
            with torch.no_grad():
                generated_ids = cls._model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )
            generated_text = cls._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = cls._processor.post_process_generation(
                generated_text, task=task_prompt, image_size=(rgb.width, rgb.height)
            )
            value = parsed.get(task_prompt, "") if isinstance(parsed, dict) else parsed
            return str(value or "").strip()
        except Exception as e:
            logger.info(f"VLM inference failed for {task_prompt}: {e}")
            return ""

    @classmethod
    def ocr(cls, image) -> str:
        """Transcribe text from an image/region crop."""
        return cls._run(image, _OCR_TASK)

    @classmethod
    def caption(cls, image) -> str:
        """Describe the visual content of an image/region crop."""
        return cls._run(image, _CAPTION_TASK)


def vlm_available() -> bool:
    return VLMEngine.available()


def vlm_ocr(image) -> str:
    return VLMEngine.ocr(image)


def vlm_caption(image) -> str:
    return VLMEngine.caption(image)
