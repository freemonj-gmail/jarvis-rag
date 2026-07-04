"""Vision-language model (VLM) escalation for the page pipeline (OCR Phase 2).

Wraps a local vision-language model — SmolVLM2 (Hugging Face,
Apache-2.0) by default — behind a small, defensive interface used by
``PageProcessor`` to:

* re-OCR regions the cheaper stages (Surya recognition / pytesseract / easyocr)
  could not read confidently, and
* optionally caption figure/photo regions so non-text visual content becomes
  searchable.

Why SmolVLM2 and not Florence-2: Florence-2's ``trust_remote_code`` model is
frozen against the transformers 4.4x era and fails to load on the transformers
5.x / huggingface-hub 1.x stack this project uses (and pinning it down conflicts
with the RAG core). SmolVLM2 is natively supported by modern transformers
(``AutoModelForImageTextToText``), needs no remote code, and runs comfortably on
Apple Silicon.

Design mirrors the other optional backends in this package:

* Nothing heavy is imported at module load. ``torch``/``transformers`` are
  imported lazily the first time the engine is actually used, and only when
  ``JARVIS_ENABLE_VLM`` is set.
* The model is loaded once and shared at class level.
* Every load/inference path is guarded; any failure marks the engine
  unavailable and the caller falls back to its OCR result. A VLM problem must
  never break extraction.

The interface is model-agnostic on purpose: point ``JARVIS_VLM_MODEL`` at any
image-text-to-text model transformers supports (other SmolVLM/Idefics
sizes, PaliGemma 2, Llama 3.2 Vision, ...) without touching the pipeline. On
Apple Silicon, ``HuggingFaceTB/SmolVLM2-500M-Video-Instruct`` is a lighter/faster
option; the 2.2B default gives better transcription fidelity.
"""

from __future__ import annotations

# Instruction prompts. General VLMs transcribe more reliably when told to copy
# text verbatim (important for IDs: license numbers, dates, addresses) rather
# than summarize what they see.
_OCR_PROMPT = (
    "Transcribe all text in this image exactly as it appears, preserving "
    "numbers, dates, punctuation, and line breaks. Output only the transcribed "
    "text with no commentary."
)
_CAPTION_PROMPT = "Describe this image in detail."

# Cap the longest edge fed to the VLM. High-megapixel inputs (e.g. 12 MP phone
# photos of an ID) get split into so many image tiles that SmolVLM/Idefics-class
# models under-generate or emit degenerate output, especially on MPS. Downscaling
# to a bounded edge keeps the tile count sane while preserving text legibility.
_MAX_IMAGE_EDGE = 2048

from jarvisrag.config import (
    JARVIS_ENABLE_VLM,
    JARVIS_VLM_MODEL,
    logger,
)


class VLMEngine:
    """Lazy, process-wide image-text-to-text VLM wrapper (SmolVLM2 by default)."""

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
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore[import-not-found]

            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                # float16 on MPS produces NaNs for some image inputs (degenerate
                # one-word / empty generations); float32 is numerically stable.
                cls._device, cls._dtype = "mps", torch.float32
            elif torch.cuda.is_available():
                cls._device, cls._dtype = "cuda", torch.bfloat16
            else:
                cls._device, cls._dtype = "cpu", torch.float32

            logger.info(f"Loading VLM '{JARVIS_VLM_MODEL}' on {cls._device} (one-time)...")
            cls._processor = AutoProcessor.from_pretrained(JARVIS_VLM_MODEL)
            cls._model = AutoModelForImageTextToText.from_pretrained(
                JARVIS_VLM_MODEL,
                torch_dtype=cls._dtype,
            )
            cls._model = cls._model.to(cls._device).eval()
            cls._available = True
            logger.info("VLM initialized for page pipeline.")
        except Exception as e:
            logger.warning(f"VLM initialization failed; page pipeline will skip VLM escalation: {e}")
            cls._available = False

    @classmethod
    def _run(cls, image, prompt: str) -> str:
        cls._ensure_loaded()
        if not cls._available:
            return ""
        try:
            import torch  # type: ignore[import-not-found]
            from PIL import Image, ImageOps

            rgb = image.convert("RGB") if getattr(image, "mode", "RGB") != "RGB" else image
            # Honour EXIF orientation: phone photos of documents are frequently
            # stored rotated, and a sideways ID is nearly unreadable to the VLM.
            rgb = ImageOps.exif_transpose(rgb)
            longest = max(rgb.width, rgb.height)
            if longest > _MAX_IMAGE_EDGE:
                scale = _MAX_IMAGE_EDGE / float(longest)
                new_size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
                rgb = rgb.resize(new_size, Image.LANCZOS)
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            chat_text = cls._processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = cls._processor(text=chat_text, images=[rgb], return_tensors="pt")
            inputs = {k: (v.to(cls._device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                generated_ids = cls._model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                )
            new_tokens = generated_ids[:, prompt_len:]
            text = cls._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
            return (text or "").strip()
        except Exception as e:
            logger.info(f"VLM inference failed: {e}")
            return ""

    @classmethod
    def ocr(cls, image) -> str:
        """Transcribe text from an image/region crop."""
        return cls._run(image, _OCR_PROMPT)

    @classmethod
    def caption(cls, image) -> str:
        """Describe the visual content of an image/region crop."""
        return cls._run(image, _CAPTION_PROMPT)


def vlm_available() -> bool:
    return VLMEngine.available()


def vlm_ocr(image) -> str:
    return VLMEngine.ocr(image)


def vlm_caption(image) -> str:
    return VLMEngine.caption(image)
