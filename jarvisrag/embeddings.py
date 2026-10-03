"""Keep Apple-GPU (MPS) memory bounded while embedding.

PyTorch's MPS backend keeps GPU resources for every distinct input shape it
runs and never releases them. Stock sentence-transformers batching pads each
batch to its own longest text, so nearly every batch is a new (batch, length)
shape and GPU memory grows until the device limit: bulk indexing stalled at
~89 GB after a few hundred chunks, and long chunks hit the limit within ~200.

Measured fix (bge-m3, MPS): pad every batch to one of a small fixed set of
lengths, shrink the batch as the length grows so each batch holds a similar
number of tokens, and split leftover chunks into power-of-two batches
(13 -> 8 + 4 + 1). Only a bounded set of shapes ever runs, so memory stays flat
(4.6 GB over 1,000 ordinary chunks, ~13 GB over 2,000 long ones, where it
previously grew past 40 GB) and embedding is faster (19.5 vs 11.6 chunks/s).
Vectors are unchanged: padding is masked out (max |diff| ~5e-7).

Installed on the model INSTANCE rather than via a subclass: the class identity
is part of every saved index signature (and of the mail index's chunk IDs), so
a subclass would force a full re-embed of every corpus for identical vectors.
"""

from __future__ import annotations

from typing import Any, List, Optional

# Batches hold at most this many padded tokens. 32 x 256 keeps the configured
# batch of 32 for ordinary chunks and shrinks it only for long ones.
_TOKENS_PER_BATCH = 32 * 256
_SHORTEST_LENGTH = 128


def _padded_length(length: int, longest: Optional[int]) -> int:
    """Smallest of 128, 192, 256, 384, 512, 768, ... that holds ``length``."""

    target = _SHORTEST_LENGTH
    while target < length:
        target = target * 3 // 2 if target & (target - 1) == 0 else target // 3 * 4
    return min(target, longest) if longest else target


def bound_mps_embedding_shapes(embed_model: Any) -> bool:
    """Install fixed-shape batching on a HuggingFaceEmbedding running on MPS.

    Returns True when installed; a model on any other device is left untouched.
    """

    sentence_model = getattr(embed_model, "_model", None)
    if sentence_model is None or not str(getattr(sentence_model, "device", "")).startswith("mps"):
        return False

    import torch

    module = sentence_model[0]
    stock_tokenize = module.tokenize

    def tokenize_to_fixed_length(texts, *args, **kwargs):
        features = stock_tokenize(texts, *args, **kwargs)
        length = features["input_ids"].shape[1]
        extra = _padded_length(length, module.max_seq_length) - length
        if extra > 0:
            pad_id = module.tokenizer.pad_token_id
            for key, value in list(features.items()):
                if torch.is_tensor(value) and value.dim() == 2:
                    fill = pad_id if key == "input_ids" else 0
                    features[key] = torch.nn.functional.pad(value, (0, extra), value=fill)
        return features

    module.tokenize = tokenize_to_fixed_length

    stock_embed = embed_model._embed

    def embed_in_fixed_batches(inputs: List[Any], prompt_name: Optional[str] = None) -> List[List[float]]:
        if len(inputs) <= 1 or not all(isinstance(item, str) for item in inputs):
            return stock_embed(inputs, prompt_name)
        token_counts = [
            len(ids)
            for ids in module.tokenizer(
                list(inputs),
                add_special_tokens=True,
                truncation=True,
                max_length=module.max_seq_length,
            )["input_ids"]
        ]
        groups: dict[int, list[int]] = {}
        for position, count in enumerate(token_counts):
            groups.setdefault(_padded_length(count, module.max_seq_length), []).append(position)

        results: list[Optional[List[float]]] = [None] * len(inputs)
        configured = embed_model.embed_batch_size
        try:
            for length, positions in groups.items():
                size = max(1, min(configured, _TOKENS_PER_BATCH // length))
                full = len(positions) - len(positions) % size
                runs = [(positions[:full], size)] if full else []
                # Split the remainder into power-of-two batches (13 -> 8 + 4 + 1)
                # so partial batches reuse a few fixed sizes without padding rows.
                start, remainder = full, len(positions) - full
                while remainder:
                    part = 1 << (remainder.bit_length() - 1)
                    runs.append((positions[start:start + part], part))
                    start, remainder = start + part, remainder - part
                for run, run_size in runs:
                    embed_model.embed_batch_size = run_size
                    vectors = stock_embed([inputs[position] for position in run], prompt_name)
                    for position, vector in zip(run, vectors):
                        results[position] = vector
        finally:
            embed_model.embed_batch_size = configured
        return results  # type: ignore[return-value]

    # Pydantic models reject unknown attributes; bypass it to override per instance.
    object.__setattr__(embed_model, "_embed", embed_in_fixed_batches)
    return True
