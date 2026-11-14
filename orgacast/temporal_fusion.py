"""Parallel local/temporal control fusion (OrgaCast, section 3.3).

The paper does not specify the temporal kernel width, reduction over frames,
dropout probability or gate parameterization. These are saved configuration
choices, not recovered hyperparameters from the published experiment.
"""

import torch
from torch import nn
import einops

from diffusers.models.attention import BasicTransformerBlock


class TemporalAttentionStream(nn.Module):
    """Attention over frames independently at each spatial position."""
    def __init__(self, channels, head_width, groups, dropout):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer = BasicTransformerBlock(
            channels, channels // head_width, head_width,
            dropout=dropout, double_self_attention=True,
        )
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, features, cross_attention_kwargs=None):
        batch, _, _, height, width = features.shape
        tokens = einops.rearrange(self.norm(features), "b c t h w -> (b h w) t c")
        tokens = self.transformer(self.proj_in(tokens), cross_attention_kwargs=cross_attention_kwargs)
        tokens = self.proj_out(tokens)
        return einops.rearrange(tokens, "(b h w) t c -> b c t h w", b=batch, h=height, w=width)


class HybridTemporalFusion(nn.Module):
    def __init__(self, channels, attention_head_dim, norm_num_groups,
                 temporal_kernel_size=3, dropout=0.1):
        super().__init__()
        if temporal_kernel_size < 3 or temporal_kernel_size % 2 != 1:
            raise ValueError("Temporal convolution needs an odd kernel of at least three frames.")
        if channels % attention_head_dim:
            raise ValueError("Attention head width must divide the feature channels.")
        self.local_conv = nn.Conv3d(
            channels, channels, (temporal_kernel_size, 1, 1),
            padding=(temporal_kernel_size // 2, 0, 0),
        )
        self.temporal_attention = TemporalAttentionStream(channels, attention_head_dim, norm_num_groups, dropout)
        self.attention_projection = nn.Conv3d(channels, channels, 1)
        self.local_weight_logit = nn.Parameter(torch.zeros(()))
        # Both control streams initially contribute zero to the pretrained U-Net.
        for projection in (self.local_conv, self.attention_projection):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, features, cross_attention_kwargs=None):
        local = self.local_conv(features)
        attention = self.attention_projection(self.temporal_attention(features, cross_attention_kwargs))
        local_weight = self.local_weight_logit.sigmoid().to(features.dtype)
        return (1 - local_weight) * attention + local_weight * local
