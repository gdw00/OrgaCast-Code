"""OrgaCast models and pipelines for fluorescence organoid forecasting."""

from .orgacast_unet import OrgaCastUNet
from .controlnet import ControlNetModel
from .controlnet_3d import ControlNetModel3D
from .data_util import metadata_normalize
from .pipeline import StableDiffusionPipeline as OrgaCastPipeline
from .pipeline_controlnet import StableDiffusionControlNetPipeline as OrgaCastControlNetPipeline

__all__ = [
    "OrgaCastUNet",
    "ControlNetModel",
    "ControlNetModel3D",
    "OrgaCastPipeline",
    "OrgaCastControlNetPipeline",
    "metadata_normalize",
]


def __getattr__(name):
    # Historical package-level imports remain available outside the public API.
    from .compat import LEGACY_EXPORTS

    if name in LEGACY_EXPORTS:
        return LEGACY_EXPORTS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
