"""Image objectives and a hard-area value with an explicit surrogate gradient."""

import math
import torch
import torch.nn.functional as F

from .masking import DEFAULT_MASK_CONFIG, foreground_masks


AREA_LOSS_VERSION = "hard_area_sigmoid_ste_v1"
DEFAULT_MORPH_TEMPERATURE = 0.02


def area_loss_configuration(temperature=DEFAULT_MORPH_TEMPERATURE):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Area surrogate temperature must be finite and positive.")
    return {
        "method": AREA_LOSS_VERSION,
        "forward": "hard_mask_area",
        "backward": "sigmoid_at_detached_low_otsu_threshold",
        "temperature": temperature,
        "temperature_units": "channel_mean_intensity_0_to_1",
        "area_units": "pixels",
        "reduction": "mean_absolute_per_sample_area_difference",
        "gaussian_sigma": DEFAULT_MASK_CONFIG.sigma,
        "gaussian_mode": "nearest",
        "gaussian_truncate": 4.0,
        "low_threshold_ratio": DEFAULT_MASK_CONFIG.hysteresis_ratio,
        "threshold_gradient": False,
        "connectivity_and_cleanup_gradient": False,
        "target_gradient": False,
        "learnable_mask_parameters": False,
        "prediction_clamp": [0, 1],
    }


def _smoothed_channel_sum(images, sigma):
    """Differentiable separable Gaussian with the hard mask's nearest padding."""
    summed = images.float().sum(dim=1, keepdim=True)
    if sigma == 0:
        return summed
    radius = int(4.0 * sigma + 0.5)
    coordinates = torch.arange(-radius, radius + 1, device=images.device, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = (kernel / kernel.sum()).float()
    # scipy.ndimage filters axis 0 then axis 1; replicate matches mode='nearest'.
    vertical = F.conv2d(F.pad(summed, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return F.conv2d(F.pad(vertical, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))


def hard_foreground_area_loss(predicted_images, target_images, temperature=DEFAULT_MORPH_TEMPERATURE, *, target_mask=None):
    """Exact hard-area error in the forward pass, sigmoid surrogate backward.

    Inputs use the dataset/decoder [-1,1] convention. Clamp only this mask
    path to [0,1]; the differentiable image objectives remain unchanged.
    The caller supplies actual target pixels, not a VAE reconstruction.

    Otsu, connectivity, cleanup and target masks are detached. Only predicted
    intensities receive the approximate gradient; the mask has no parameters.
    This is an engineering extension, not an exact hard-segmentation derivative.
    """
    area_loss_configuration(temperature)
    if predicted_images.shape != target_images.shape:
        raise ValueError("Predicted and target images must have matching shapes.")
    # Keep this branch in float32 even if the surrounding training uses autocast.
    with torch.autocast(device_type=predicted_images.device.type, enabled=False):
        if not torch.isfinite(predicted_images).all() or not torch.isfinite(target_images).all():
            raise ValueError("Area loss requires finite predicted and target images.")
        predicted = (predicted_images.float() / 2 + 0.5).clamp(0, 1)
        target = (target_images.detach().to(predicted.device, dtype=torch.float32) / 2 + 0.5).clamp(0, 1)
        hard_masks, thresholds = foreground_masks(predicted, return_thresholds=True)
        hard_area = hard_masks.sum(dim=(1, 2, 3))
        if target_mask is None:
            target_area = foreground_masks(target).sum(dim=(1, 2, 3))
        else:
            expected = (target.shape[0], 1, *target.shape[-2:])
            if target_mask.dtype != torch.bool or tuple(target_mask.shape) != expected:
                raise ValueError("Cached target masks must be boolean tensors of shape [B, 1, H, W].")
            target_area = target_mask.detach().to(device=predicted.device, dtype=torch.float32).sum(dim=(1, 2, 3))
        if not torch.is_grad_enabled() or not predicted.requires_grad:
            return F.l1_loss(hard_area, target_area)
        smoothed = _smoothed_channel_sum(predicted, DEFAULT_MASK_CONFIG.sigma)
        low = thresholds * DEFAULT_MASK_CONFIG.hysteresis_ratio
        # Divide both intensities and thresholds by three to give tau [0,1] units.
        soft_area = torch.sigmoid((smoothed - low) / (3 * temperature)).sum(dim=(1, 2, 3))
        train_area = hard_area + (soft_area - soft_area.detach())
        return F.l1_loss(train_area, target_area)


def predicted_clean_latents(noisy_latents, prediction, timesteps, scheduler):
    alpha = scheduler.alphas_cumprod.to(noisy_latents.device)[timesteps].float()
    alpha = alpha.view(-1, 1, 1, 1)
    noisy, prediction = noisy_latents.float(), prediction.float()
    if scheduler.config.prediction_type == "epsilon":
        return (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt().clamp_min(1e-8)
    if scheduler.config.prediction_type == "v_prediction":
        return alpha.sqrt() * noisy - (1 - alpha).sqrt() * prediction
    raise ValueError(f"Unsupported diffusion prediction type: {scheduler.config.prediction_type}")


def image_objectives(vae, clean_latents, target_images, perceptual_model, lpips_weight):
    # Freeze VAE parameters, not this decode operation: gradients must reach
    # the predicted latents, control branch and target metadata adapters.
    predicted = vae.decode(clean_latents.to(dtype=vae.dtype) / vae.config.scaling_factor).sample
    target = target_images.to(device=predicted.device, dtype=predicted.dtype)
    pixel_mse = F.mse_loss(predicted.float(), target.float())
    perceptual = predicted.new_zeros(())
    if lpips_weight > 0:
        # The LPIPS network is frozen; the predicted-image path stays differentiable.
        perceptual = perceptual_model(predicted.float().clamp(-1, 1), target.float()).mean()
    return predicted, pixel_mse, perceptual
