"""Image metrics with explicit ranges and retained per-sample records."""

import json
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def image_metrics(prediction, target, lpips_model):
    pred_np = prediction.detach().float().cpu().permute(1, 2, 0).numpy()
    target_np = target.detach().float().cpu().permute(1, 2, 0).numpy()
    result = {
        "ssim": float(structural_similarity(target_np, pred_np, data_range=1.0, channel_axis=-1, win_size=11)),
        "psnr": float(peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)),
        "lpips": float(lpips_model(prediction[None].float() * 2 - 1, target[None].float() * 2 - 1).item()),
    }
    result["channels"] = {
        name: {
            "ssim": float(structural_similarity(target_np[..., index], pred_np[..., index], data_range=1.0, win_size=11)),
            "psnr": float(peak_signal_noise_ratio(target_np[..., index], pred_np[..., index], data_range=1.0)),
        } for index, name in enumerate(("CH1", "CH2", "CH3"))
    }
    return result


def save_metric_records(path, records, step=None):
    groups = {}
    for row in records:
        condition = row["sample_key"].rsplit("_", 1)[0]
        groups.setdefault(condition, []).append(row)
    def average(rows):
        return {name: float(np.mean([row[name] for row in rows])) for name in ("ssim", "psnr", "lpips")}
    output = {
        "step": step, "image_range": [0, 1], "lpips_input_range": [-1, 1],
        "ssim_window": 11, "channel_order": ["CH1", "CH2", "CH3"],
        "samples": records, "mean": average(records) if records else {},
        "conditions": {key: dict(count=len(rows), **average(rows)) for key, rows in groups.items()},
    }
    Path(path).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    return output["mean"]
