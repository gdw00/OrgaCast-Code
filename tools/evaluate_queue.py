"""Evaluate immutable training snapshots on a separate, visible GPU."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import lpips
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, CLIPTextModel

from orgacast import OrgaCastUNet, ControlNetModel3D
from orgacast.cell_data import MultiConditionTimeSeriesDataset, InferenceOnlyDataset, collate_cells
from orgacast.data_cache import CachedCellDataset
from orgacast.evaluation_queue import pending_jobs, validate_snapshot, discard_completed_snapshots
from orgacast.training_state import write_json
from orgacast.text_conditioning import caption_audit
from train_controlnet import log_validation, generate_with_uncertainty


def evaluation_accelerator(*, cpu=False):
    accelerator = Accelerator(cpu=cpu, mixed_precision="no", log_with=None)
    # Accelerate 0.21 initializes `trackers` here even with logging disabled.
    accelerator.init_trackers("orgacast_evaluation")
    return accelerator


def assert_same_control_configuration(previous, current):
    # from_pretrained records its input path; resuming changes this provenance
    # field without changing any architecture or conditioning setting.
    previous = {key: value for key, value in previous.items() if key != "_name_or_path"}
    current = {key: value for key, value in current.items() if key != "_name_or_path"}
    if previous != current:
        raise ValueError("ControlNet configuration changed inside the evaluation queue.")


def load_snapshot(path, unet, controlnet=None):
    """Copy a matched snapshot into existing inference modules."""
    job = validate_snapshot(path)
    path = Path(path)
    if controlnet is None:
        controlnet = ControlNetModel3D.from_pretrained(path / "controlnet", local_files_only=True)
    else:
        state = torch.load(path / "controlnet/diffusion_pytorch_model.bin", map_location="cpu")
        controlnet.load_state_dict(state, strict=True)
    adapters = torch.load(path / "target_metadata.bin", map_location="cpu")
    unet.metadata_embedding.load_state_dict(adapters, strict=True)
    return controlnet, job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    cli = parser.parse_args()
    output = Path(cli.output_dir).resolve()
    lock = (output / "evaluation_worker.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = json.loads((output / "evaluation_queue/config.json").read_text())
    args = SimpleNamespace(**request["args"])
    os.chdir(request["working_directory"])
    status = {"state": "starting", "pid": os.getpid(), "gpu": request["evaluation_gpu"], "completed_steps": []}

    def update(**values):
        status.update(values, updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(output / "evaluation_status.json", status)

    update()
    try:
        accelerator = evaluation_accelerator()
        if accelerator.device.type != "cuda":
            raise RuntimeError("Asynchronous evaluation requires its assigned CUDA GPU.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.pretrained_model_name_or_path,
                                                subfolder=None if args.tokenizer_name else "tokenizer",
                                                revision=args.revision, use_fast=False, local_files_only=True)
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", local_files_only=True)
        vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True)
        unet = OrgaCastUNet.from_pretrained(output / "evaluation_assets/unet", local_files_only=True)
        perceptual = lpips.LPIPS(net="vgg")
        for model in (text_encoder, vae, unet, perceptual):
            model.requires_grad_(False).eval().to(accelerator.device, dtype=torch.float32)
        scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
        dataset = MultiConditionTimeSeriesDataset(args.train_data_dir, tokenizer=tokenizer, args=args)
        text_audit = caption_audit(tokenizer, args, dataset.caption_texts)
        training_audit = output / "text_conditioning_audit.json"
        if training_audit.exists() and json.loads(training_audit.read_text()) != text_audit:
            raise ValueError("Training and evaluation text encoding disagree.")
        write_json(output / "evaluation_text_conditioning_audit.json", text_audit)
        if args.cache_preprocessed_data:
            dataset = CachedCellDataset(dataset, cache_target_masks=args.cache_target_masks)
        lookup = {key: index for index, key in enumerate(dataset.samples)}
        subset = Subset(dataset, [lookup[key] for key in request["split"]["validation"]])
        loader = DataLoader(subset, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_cells,
                            generator=torch.Generator().manual_seed(args.seed + 1), num_workers=0)
        controlnet, inference_loader, control_config = None, None, None
        result_dir = output / "evaluation_results"
        result_dir.mkdir(exist_ok=True)
        update(state="running")
        while True:
            discard_completed_snapshots(output)
            status["completed_steps"] = sorted(int(p.stem[3:]) for p in result_dir.glob("gs_*.json"))
            for job_file in pending_jobs(output):
                job = validate_snapshot(job_file.parent)
                completed = result_dir / f"gs_{job['step']}.json"
                if completed.exists():
                    if job["step"] not in status["completed_steps"]:
                        status["completed_steps"].append(job["step"])
                    continue
                start = time.perf_counter()
                update(state="evaluating", step=job["step"], stage="loading_snapshot")
                configuration = json.loads((job_file.parent / "controlnet/config.json").read_text())
                if control_config is not None:
                    assert_same_control_configuration(control_config, configuration)
                control_config = configuration
                controlnet, job = load_snapshot(job_file.parent, unet, controlnet)
                controlnet.requires_grad_(False).eval().to(accelerator.device, dtype=torch.float32)
                if job["final"] and args.inference_data_dir and inference_loader is None:
                    inference = InferenceOnlyDataset(args.inference_data_dir, tokenizer=tokenizer, args=args)
                    if list(inference.samples) != request["split"]["inference"]:
                        raise ValueError("Inference-only sample IDs changed after training started.")
                    if args.cache_preprocessed_data:
                        inference = CachedCellDataset(inference)
                    inference_loader = DataLoader(inference, batch_size=args.train_batch_size, shuffle=False,
                                                  collate_fn=collate_cells, generator=torch.Generator().manual_seed(args.seed + 2))
                update(stage="validation_and_final_inference" if job["final"] else "validation")
                metrics = log_validation(
                    vae, text_encoder, tokenizer, unet, controlnet, scheduler, args, accelerator,
                    torch.float32, job["step"], loader, perceptual,
                    inference_dataloader=inference_loader if job["final"] else None,
                )
                if job["final"] and args.run_uncertainty_analysis:
                    update(stage="uncertainty", mc_samples=args.mc_samples)
                    for batch in loader:
                        generate_with_uncertainty(
                            vae, text_encoder, unet, controlnet, scheduler, accelerator,
                            torch.float32, args, batch, output / args.uncertainty_output_dir,
                            num_mc_samples=args.mc_samples,
                        )
                write_json(completed, {
                    "state": "completed", "step": job["step"], "final": job["final"],
                    "metrics": metrics, "wall_seconds": time.perf_counter() - start,
                    "inference_only": job["final"] and bool(args.inference_data_dir),
                    "mc_samples": args.mc_samples if job["final"] and args.run_uncertainty_analysis else 0,
                    "snapshot": str(job_file.parent),
                    "snapshot_files": job["files"], "snapshot_retained": False,
                })
                discard_completed_snapshots(output)
                status["completed_steps"].append(job["step"])
                update(state="running", stage="waiting")
            producer = json.loads((output / "evaluation_queue/producer.json").read_text())
            if producer["state"] in {"completed", "paused", "failed"}:
                # A producer may finish while the previous directory scan runs.
                remaining = [p for p in pending_jobs(output)
                             if not (result_dir / f"gs_{validate_snapshot(p.parent)['step']}.json").exists()]
                if remaining:
                    continue
                if producer["state"] == "completed" and not (result_dir / f"gs_{args.max_train_steps}.json").exists():
                    raise RuntimeError("Training finished without a final evaluation snapshot.")
                update(state=producer["state"], stage="finished")
                return
            try:
                os.kill(producer["pid"], 0)
            except ProcessLookupError:
                raise RuntimeError("Training exited without closing the evaluation queue.")
            time.sleep(2)
    except Exception as error:
        update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
