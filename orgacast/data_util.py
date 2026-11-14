"""Four raw cell metadata fields and a seven-field historical weight adapter.

New experiments use condition ID, DY indicator, DAPT indicator and day. The
seven-field path preserves old positional scaling for historical model loading.
"""
import torch


def metadata_normalize(metadata, base_feature0=180, base_feature1=90, base_feature4=1980, max_feature2=1., scale=1000):
    if len(metadata) == 4:
        # The paper specifies four semantic scalars, but no numeric scaling.
        # New models use raw fields with independently learned sinusoidal/MLP
        # embeddings; this choice is recorded in their saved configuration.
        return torch.as_tensor(metadata, dtype=torch.float32).clone()
    if len(metadata) == 0:
        return torch.empty(0, dtype=torch.float32)
    if len(metadata) != 7:
        raise ValueError("Expected four cell fields or seven historical compatibility fields.")
    feature0, feature1, feature2, feature3, feature4, feature5, feature6 = metadata
    feature0 = feature0 / (180 + base_feature0) * scale
    feature1 = feature1 / (90 + base_feature1) * scale
    feature2 = feature2 / max_feature2 * scale
    feature3 = feature3 * scale
    feature4 = feature4 / (2100 - base_feature4) * scale
    feature5 = feature5 / 12 * scale
    feature6 = feature6 / 31 * scale
    return torch.tensor([feature0, feature1, feature2, feature3, feature4, feature5, feature6])
