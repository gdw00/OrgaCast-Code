"""Shared CLIP encoding for complete captions and historical truncation.

Each chunk uses CLIP's existing positions. Concatenation exposes every chunk to
cross-attention; CLIP itself does not attend across chunk boundaries. Padding
uses the tokenizer's normal EOS/pad convention, including empty trailing chunks.
"""

import math

import torch


def text_conditioning_configuration(args):
    mode = getattr(args, "text_encoding_mode", "truncate")
    chunks = getattr(args, "text_max_chunks", 3)
    if mode not in {"truncate", "chunked"} or type(chunks) is not int or chunks < 1:
        raise ValueError("Text mode must be truncate/chunked and text_max_chunks a positive integer.")
    return {
        "version": "clip_chunked_concat_v1" if mode == "chunked" else "clip_truncate_v1",
        "mode": mode,
        "max_chunks": chunks if mode == "chunked" else 1,
        "chunk_special_tokens": "retain_bos_eos",
        "padding": "fixed_chunks_with_tokenizer_pad",
        "overflow": "error" if mode == "chunked" else "truncate",
        "frozen_embedding_cache": True,
        "caption_version": "original_group_captions_v1",
    }


def caption_token_ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False, truncation=False,
                     return_attention_mask=False, verbose=False)["input_ids"]


def tokenize_caption(tokenizer, text, args):
    config = text_conditioning_configuration(args)
    length = tokenizer.model_max_length
    if config["mode"] == "truncate":
        return tokenizer(text, max_length=length, padding="max_length", truncation=True,
                         return_tensors="pt").input_ids.squeeze(0)
    capacity = length - 2
    if capacity < 1 or length > 10000:
        raise ValueError("Chunked CLIP encoding requires a finite tokenizer context length.")
    if any(value is None for value in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)):
        raise ValueError("Chunked CLIP encoding requires BOS, EOS and padding tokens.")
    ids = caption_token_ids(tokenizer, text)
    if len(ids) > capacity * config["max_chunks"]:
        raise ValueError(
            f"Caption has {len(ids)} content tokens; {config['max_chunks']} chunks hold "
            f"{capacity * config['max_chunks']}. Increase --text_max_chunks; no text was truncated."
        )
    rows = []
    for start in range(0, capacity * config["max_chunks"], capacity):
        content = ids[start:start + capacity]
        rows.append([tokenizer.bos_token_id, *content, tokenizer.eos_token_id]
                    + [tokenizer.pad_token_id] * (capacity - len(content)))
    return torch.tensor(rows, dtype=torch.long)


def _encode(text_encoder, input_ids):
    parameter = next(text_encoder.parameters())
    batch = input_ids.shape[0]
    flat = input_ids.reshape(-1, input_ids.shape[-1]).to(parameter.device)
    hidden = text_encoder(flat)[0]
    return hidden.reshape(batch, -1, hidden.shape[-1])


def encode_text_conditioning(text_encoder, input_ids, *, cache=True):
    """Accept [B,L] or [B,K,L]; cache only a frozen encoder in evaluation mode.

    Cache entries are bound to the encoder's parameter identities, versions,
    device and dtype. Loading different weights cannot reuse stale embeddings.
    The cache is process-local and is never saved as model/checkpoint state.
    """
    if input_ids.ndim not in (2, 3) or input_ids.shape[0] == 0:
        raise ValueError("Text IDs must be a nonempty [B,L] or [B,K,L] tensor.")
    parameters = tuple(text_encoder.parameters())
    frozen = not text_encoder.training and all(not p.requires_grad for p in parameters)
    if not cache or not frozen:
        return _encode(text_encoder, input_ids)
    signature = tuple((id(p), p._version, p.device, p.dtype) for p in parameters)
    if getattr(text_encoder, "_orgacast_text_cache_signature", None) != signature:
        text_encoder._orgacast_text_cache_signature = signature
        text_encoder._orgacast_text_cache = {}
    embeddings = text_encoder._orgacast_text_cache
    rows = input_ids.detach().cpu()
    keys = [(tuple(row.shape), tuple(row.reshape(-1).tolist())) for row in rows]
    if len(set(keys)) > 128:
        with torch.no_grad():
            return _encode(text_encoder, input_ids)
    if len(embeddings) + len(set(keys) - embeddings.keys()) > 128:
        embeddings.clear()
    missing = {}
    for key, row in zip(keys, rows):
        if key not in embeddings:
            missing.setdefault(key, row)
    if missing:
        with torch.no_grad():
            encoded = _encode(text_encoder, torch.stack(list(missing.values())))
        for key, value in zip(missing, encoded):
            embeddings[key] = value.detach().clone()
    return torch.stack([embeddings[key] for key in keys])


def caption_audit(tokenizer, args, captions):
    """Record original and actually encoded text for each named caption."""
    configuration = text_conditioning_configuration(args)
    capacity = tokenizer.model_max_length - 2
    records = []
    for name, text in captions.items():
        ids = caption_token_ids(tokenizer, text)
        encoded = tokenize_caption(tokenizer, text, args)
        truncated = configuration["mode"] == "truncate" and len(ids) > capacity
        records.append({
            "condition": name, "full_prompt": text, "content_token_count": len(ids),
            "token_count_with_specials": len(ids) + 2,
            "required_chunks": max(1, math.ceil(len(ids) / capacity)),
            "input_ids_shape": list(encoded.shape), "truncated": truncated,
            "retained_content_tokens": min(len(ids), capacity) if truncated else len(ids),
            "decoded_chunks": [tokenizer.decode(row, skip_special_tokens=True)
                               for row in encoded.reshape(-1, tokenizer.model_max_length).tolist()],
        })
    return dict(configuration, model_context_length=tokenizer.model_max_length,
                content_capacity_per_chunk=capacity, records=records)
