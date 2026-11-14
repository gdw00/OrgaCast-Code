"""Historical class names for old callers.

Load existing U-Net checkpoints with ``OrgaCastUNet.from_pretrained(path)``.
The bundled ModelMixin accepts their old ``_class_name`` and unchanged state
dictionary keys. Saving the loaded model writes the canonical OrgaCast name.
"""

from .orgacast_unet import OrgaCastUNet

LEGACY_EXPORTS = {"SatUNet": OrgaCastUNet}
