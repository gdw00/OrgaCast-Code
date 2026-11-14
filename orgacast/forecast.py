"""One deterministic forecast path for component checkpoints and cell batches."""

import torch
import einops
from diffusers import UniPCMultistepScheduler
from .paper_training import target_metadata_kwargs, control_scale
from .text_conditioning import encode_text_conditioning


@torch.no_grad()
def forecast_batch(unet, controlnet, vae, text_encoder, scheduler, batch, args, generator):
    device, dtype = next(unet.parameters()).device, next(unet.parameters()).dtype
    hidden = encode_text_conditioning(text_encoder, batch["input_ids"]).to(dtype=dtype)
    images = einops.rearrange(batch["conditioning_pixel_values"].to(device=device, dtype=dtype), "b t c h w -> b c t h w")
    metadata = batch["metadata"].to(device=device, dtype=dtype)
    history = batch["conditioning_metadata"].to(device=device, dtype=dtype)
    scales = getattr(args, "controlnet_scales", None)
    if scales is not None:
        images = images * torch.tensor(scales, device=device, dtype=dtype).view(1, 1, -1, 1, 1)
    latents = torch.randn(
        (images.shape[0], unet.config.in_channels, args.resolution // 8, args.resolution // 8),
        generator=generator, device=device, dtype=dtype,
    )
    sampler = UniPCMultistepScheduler.from_config(scheduler.config)
    sampler.set_timesteps(args.num_inference_steps)
    routed = target_metadata_kwargs(controlnet, unet, metadata)
    for timestep in sampler.timesteps:
        residuals, middle = controlnet(
            latents, timestep, encoder_hidden_states=hidden, controlnet_cond=images,
            metadata=metadata, cond_metadata=history, conditioning_scale=control_scale(args),
            return_dict=False, **routed,
        )
        prediction = unet(
            latents, timestep, encoder_hidden_states=hidden, metadata=metadata,
            down_block_additional_residuals=residuals, mid_block_additional_residual=middle,
        ).sample
        latents = sampler.step(prediction, timestep, latents).prev_sample
    decoded = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    return (decoded / 2 + 0.5).clamp(0, 1)
