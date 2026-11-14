"""Process-local cache of deterministic cell preprocessing and target masks."""

import torch
from torch.utils.data import Dataset

from .masking import foreground_masks, mask_configuration

CACHE_VERSION = "preprocessed_cells_and_fixed_masks_v1"


class CachedCellDataset(Dataset):
    """Materialize exact dataset tensors once; each access returns owned copies.

    This cache is intentionally rebuilt for each run. It cannot silently reuse
    stale transforms, source images, ablations, or mask settings from disk.
    No latents or predictions are cached, so VAE sampling stays stochastic.
    """

    def __init__(self, dataset, *, cache_target_masks=False):
        self.samples = list(dataset.samples)
        self.rows = []
        self.cache_bytes = 0
        self.mask_config = mask_configuration() if cache_target_masks else None
        for index, key in enumerate(self.samples):
            row = dataset[index]
            if row is None or row["sample_key"] != key:
                raise ValueError(f"Invalid cached cell sequence: {key}")
            row = {name: value.detach().cpu().clone() if torch.is_tensor(value) else value
                   for name, value in row.items()}
            if cache_target_masks and "pixel_values" in row:
                image = (row["pixel_values"].float().unsqueeze(0) / 2 + 0.5).clamp(0, 1)
                row["target_foreground_mask"] = foreground_masks(image)[0].bool()
            self.cache_bytes += sum(value.numel() * value.element_size()
                                    for value in row.values() if torch.is_tensor(value))
            self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {name: value.clone() if torch.is_tensor(value) else value
                for name, value in self.rows[index].items()}
