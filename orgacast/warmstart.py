"""Explicit, audited transfer from the historical seven-field control branch.

This is a new-stage initialization, never a training-state resume. Temporal
attention and metadata changed semantics and are deliberately not transplanted.
"""

import hashlib
import json
from pathlib import Path

import torch

from .temporal_fusion import HybridTemporalFusion


TRANSFER_VERSION = "legacy_spatial_local_projection_v1"
SPATIAL_PREFIXES = (
    "conv_in.", "time_embedding.", "controlnet_cond_embedding.",
    "down_blocks.", "mid_block.",
)


def transfer_legacy_state(controlnet, source, source_config, num_frames):
    """Copy shared tensors and embed old 1x1 heads in the new temporal kernel.

    For identical encoder features, the initial local stream reproduces the
    legacy *un-attended* frame-summed projection. The complete old network is
    not equivalent: its temporal attention and metadata are not carried over.
    All checks happen before modifying the destination model.
    """
    if type(num_frames) is not int or num_frames < 1:
        raise ValueError("A positive observed frame count is required.")
    if source_config.get("num_metadata") != 7 or source_config.get("temporal_fusion_mode", "legacy") != "legacy":
        raise ValueError("Warm start requires a historical seven-field legacy ControlNet.")
    if (controlnet.config.temporal_fusion_mode != "parallel"
            or controlnet.config.metadata_conditioning != "independent"
            or controlnet.config.num_metadata != 4):
        raise ValueError("Destination must use parallel fusion and independent four-field metadata.")
    destination = controlnet.state_dict()
    transfers, copied, mapped, used = {}, [], [], set()
    for name, tensor in destination.items():
        if name.startswith(SPATIAL_PREFIXES):
            if name not in source or source[name].shape != tensor.shape:
                raise ValueError(f"Incompatible spatial tensor: {name}")
            transfers[name] = source[name]
            copied.append(name)
            used.add(name)
    heads = [(f"controlnet_down_blocks.{i}", block)
             for i, block in enumerate(controlnet.controlnet_down_blocks)]
    heads.append(("controlnet_mid_block", controlnet.controlnet_mid_block))
    expected_old_heads = {f"{prefix}.{part}" for prefix, _ in heads for part in ("weight", "bias")}
    actual_old_heads = {name for name in source if name.startswith(("controlnet_down_blocks.", "controlnet_mid_block."))}
    if actual_old_heads != expected_old_heads:
        raise ValueError("Legacy control projection count or layout differs from the destination.")
    for prefix, block in heads:
        if not isinstance(block, HybridTemporalFusion):
            raise ValueError(f"Unsupported fusion module: {prefix}")
        weight, bias = source[f"{prefix}.weight"], source[f"{prefix}.bias"]
        local = block.local_conv
        expected = (local.out_channels, local.in_channels, 1, 1, 1)
        if tuple(weight.shape) != expected or tuple(bias.shape) != (local.out_channels,):
            raise ValueError(f"Incompatible legacy projection: {prefix}")
        # Default gate is 0.5 and the new reduction is mean; the old reduction
        # was sum. Compensate both factors, including the additive bias.
        local_weight = float(block.local_weight_logit.detach().sigmoid())
        factor = num_frames / local_weight
        kernel = torch.zeros_like(local.weight, device="cpu")
        kernel[:, :, kernel.shape[2] // 2, 0, 0] = weight[:, :, 0, 0, 0] * factor
        transfers[f"{prefix}.local_conv.weight"] = kernel
        transfers[f"{prefix}.local_conv.bias"] = bias * factor
        # New attention starts with zero contribution, not untrained noise.
        transfers[f"{prefix}.attention_projection.weight"] = torch.zeros_like(block.attention_projection.weight, device="cpu")
        transfers[f"{prefix}.attention_projection.bias"] = torch.zeros_like(block.attention_projection.bias, device="cpu")
        used.update([f"{prefix}.weight", f"{prefix}.bias"])
        mapped.append({"source": prefix, "destination": prefix + ".local_conv",
                       "temporal_center": kernel.shape[2] // 2, "scale": factor})
    skipped = sorted(set(source) - used)
    if any(not name.startswith(("metadata_embedding.", "temporal_blocks.", "mixing_attns.")) for name in skipped):
        raise ValueError("Unrecognized legacy tensors; refusing an unaudited partial transfer.")
    if not copied:
        raise ValueError("No compatible spatial weights found.")
    if any(not torch.isfinite(value).all() for value in transfers.values()):
        raise ValueError("Legacy transfer contains nonfinite values.")
    with torch.no_grad():
        for name, value in transfers.items():
            destination[name].copy_(value)
    source_parameters = sum(value.numel() for value in source.values())
    used_parameters = sum(source[name].numel() for name in used)
    return {
        "transfer_version": TRANSFER_VERSION, "observed_frames": num_frames,
        "copied_tensors": copied, "mapped_local_projections": mapped,
        "skipped_source_tensors": skipped,
        "source_parameter_values": source_parameters,
        "reused_source_parameter_values": used_parameters,
        "reused_source_fraction": used_parameters / source_parameters,
        "new_metadata_policy": "four raw fields; independent new target/history adapters",
        "new_temporal_policy": "parallel fusion; new attention; mapped local projections",
        "equivalence": "Only the un-attended local projection is preserved for identical encoder features. Full legacy predictions are not preserved.",
    }


def initialize_from_legacy_controlnet(controlnet, directory, num_frames):
    directory = Path(directory).resolve()
    config_file = directory / "config.json"
    weights_file = directory / "diffusion_pytorch_model.bin"
    if not config_file.is_file() or not weights_file.is_file():
        raise FileNotFoundError("Legacy transfer needs local config.json and diffusion_pytorch_model.bin.")
    digest = hashlib.sha256()
    with weights_file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    config = json.loads(config_file.read_text())
    source = torch.load(weights_file, map_location="cpu")
    report = transfer_legacy_state(controlnet, source, config, num_frames)
    report.update(source_directory=str(directory), source_weights_sha256=digest.hexdigest(),
                  source_config_sha256=hashlib.sha256(config_file.read_bytes()).hexdigest(),
                  source_config=config)
    return report
