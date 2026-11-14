"""Shared, non-trainable foreground masks for fluorescence images.

The paper uses channel SUM followed by Otsu. This selected extension smooths
the sum, grows Otsu seeds into connected lower-intensity regions, and removes
small fragments. It is the hard-mask candidate reviewed on real cell images.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from skimage.filters import apply_hysteresis_threshold, gaussian, threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects


MASK_VERSION = "sum_otsu_hysteresis_v1"


@dataclass(frozen=True)
class ForegroundMaskConfig:
    # Pixel units: selected at 256 x 256, not automatically scaled with resolution.
    sigma: float = 1.0
    hysteresis_ratio: float = 0.5
    closing_radius: int = 1
    min_size: int = 64
    hole_size: int = 64

    def __post_init__(self):
        if not math.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("Mask sigma must be finite and nonnegative.")
        if not math.isfinite(self.hysteresis_ratio) or not 0 < self.hysteresis_ratio < 1:
            raise ValueError("Mask hysteresis ratio must be between 0 and 1.")
        for name in ("closing_radius", "min_size", "hole_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Mask {name} must be a nonnegative integer.")


DEFAULT_MASK_CONFIG = ForegroundMaskConfig()


def mask_configuration(config=DEFAULT_MASK_CONFIG):
    """Serializable algorithm/settings record for outputs and resume validation."""
    return {
        "method": MASK_VERSION,
        **asdict(config),
        "input_range": [0, 1],
        "channel_reduction": "sum",
        "threshold_histogram_bins": 256,
        "hysteresis_neighbors": 4,
        "cleanup_neighbors": 8,
        "cleanup_order": ["binary_closing", "remove_small_objects", "remove_small_holes"],
        "parameter_reference_resolution": [256, 256],
        "automatically_scale_pixel_parameters": False,
        "largest_component_only": False,
        "fill_all_holes": False,
        "differentiable": False,
    }


def foreground_mask_numpy(channels, config=DEFAULT_MASK_CONFIG, *, return_threshold=False):
    """Return a boolean HW mask from finite CHW fluorescence in [0,1].

    Optionally return the detached high Otsu threshold with the mask.
    No per-image or per-channel rescaling is performed here. A constant image
    has no above-Otsu seed and produces an empty mask. Cleanup never selects
    only the largest component or deliberately fills every dark cavity.
    """
    channels = np.asarray(channels, dtype=np.float32)
    if channels.ndim != 3 or channels.shape[0] != 3 or min(channels.shape[1:]) < 1:
        raise ValueError("Expected nonempty fluorescence channels shaped [3,H,W].")
    if not np.isfinite(channels).all() or channels.min() < 0 or channels.max() > 1:
        raise ValueError("Expected finite fluorescence values in [0,1].")
    summed = channels.sum(axis=0)
    smoothed = gaussian(summed, sigma=config.sigma, preserve_range=True)
    threshold = threshold_otsu(smoothed, nbins=256)
    mask = apply_hysteresis_threshold(smoothed, config.hysteresis_ratio * threshold, threshold)
    if config.closing_radius:
        mask = binary_closing(mask, footprint=disk(config.closing_radius))
    mask = remove_small_objects(mask, min_size=config.min_size, connectivity=2)
    # Filling a tiny all-background image must not manufacture foreground.
    if mask.any():
        mask = remove_small_holes(mask, area_threshold=config.hole_size, connectivity=2)
    return (mask, float(threshold)) if return_threshold else mask


@torch.no_grad()
def foreground_masks(images, config=DEFAULT_MASK_CONFIG, *, return_thresholds=False):
    """Return detached float32 [B,1,H,W] masks on the input tensor's device.

    Input must be [B,3,H,W] in [0,1]. Computation uses float32 CPU arrays,
    matching the reviewed diagnostic, including for half-precision inputs.
    With return_thresholds=True, also return detached [B,1,1,1] high thresholds
    in channel-SUM units. The binary mask computation is identical either way.
    """
    if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[0], *images.shape[2:]) < 1:
        raise ValueError("Expected a nonempty image batch shaped [B,3,H,W].")
    values = images.detach().to(device="cpu", dtype=torch.float32).numpy()
    results = [foreground_mask_numpy(image, config, return_threshold=True) for image in values]
    masks = torch.from_numpy(np.stack([row[0] for row in results])[:, None]).to(
        device=images.device, dtype=torch.float32,
    )
    if return_thresholds:
        thresholds = torch.tensor([row[1] for row in results], device=images.device, dtype=torch.float32)
        return masks, thresholds.view(-1, 1, 1, 1)
    return masks
