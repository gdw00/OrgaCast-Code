"""Explicit training policy and metadata routing for the paper-aligned core."""

import torch
from torch import nn
from diffusers.models.embeddings import TimestepEmbedding

METHOD_VERSION = "paper_core_v3_area_ste"
METADATA_VERSION = "cell_four_fields_raw_v1"


def unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def initialize_cell_metadata(unet):
    """Retain pretrained spatial weights; initialize four new target adapters.

    A seven-field historical embedding has different semantics and is not
    silently truncated or reused with the new unscaled four-field inputs.
    """
    if getattr(unet.config, "metadata_semantics", None) == METADATA_VERSION:
        if len(unet.metadata_embedding) != 4:
            raise ValueError("Invalid four-field metadata checkpoint.")
        return
    unet.metadata_embedding = nn.ModuleList([
        TimestepEmbedding(
            unet.time_embedding.linear_1.in_features,
            unet.time_embedding.linear_2.out_features,
        ) for _ in range(4)
    ])
    for embedding in unet.metadata_embedding:
        nn.init.zeros_(embedding.linear_2.weight)
        nn.init.zeros_(embedding.linear_2.bias)
    unet.num_metadata = 4
    unet.register_to_config(num_metadata=4, use_metadata=True, metadata_semantics=METADATA_VERSION)


def set_training_policy(unet, controlnet, vae, text_encoder, perceptual_model):
    """Freeze the pretrained backbone; train control and new target adapters."""
    for model in (unet, vae, text_encoder, perceptual_model):
        model.requires_grad_(False)
        model.eval()
    if unet.metadata_embedding is not None:
        unet.metadata_embedding.requires_grad_(True)
        unet.metadata_embedding.train()
    controlnet.requires_grad_(True)
    controlnet.train()


def restore_training_modes(unet, controlnet, vae, text_encoder):
    unwrap(unet).eval()
    if unwrap(unet).metadata_embedding is not None:
        unwrap(unet).metadata_embedding.train()
    controlnet.train()
    vae.eval()
    text_encoder.eval()


def target_metadata_kwargs(controlnet, unet, metadata):
    """Route the same target encoder to both branches; history has its own MLPs."""
    if getattr(getattr(unwrap(controlnet), "config", None), "metadata_conditioning", "legacy_shared") == "independent":
        return {"target_metadata_embedding": unwrap(unet).encode_metadata(metadata)}
    return {}


def trainable_parameters(*models):
    seen = set()
    for model in models:
        for parameter in model.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter


def control_scale(args):
    return 0.0 if getattr(args, "conditioning_ablation", "none") == "images" else 1.0
