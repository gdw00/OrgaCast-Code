"""Immutable evaluation snapshots and an independent evaluation process."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import torch

from .training_state import write_json

RUNTIME_VERSION = "cached_data_async_evaluation_v1"


def pending_jobs(directory):
    return sorted((Path(directory) / "evaluation_queue/jobs").glob("gs_*/job.json"))


def discard_completed_snapshots(directory):
    """Reclaim temporary evaluation weights only after results are committed."""
    directory = Path(directory)
    jobs = directory / "evaluation_queue/jobs"
    if not jobs.is_dir():
        return
    for path in list(jobs.iterdir()):
        match = re.fullmatch(r"(?:\.retired_)?gs_(\d+)", path.name)
        if not match or not path.is_dir() or path.is_symlink():
            continue
        marker = directory / "evaluation_results" / f"gs_{int(match[1])}.json"
        if not marker.is_file():
            continue
        result = json.loads(marker.read_text())
        if result.get("state") != "completed" or result.get("step") != int(match[1]):
            continue
        retired = jobs / f".retired_gs_{int(match[1]):06d}"
        if path != retired:
            path.rename(retired)
        shutil.rmtree(retired)


def publish_evaluation_snapshot(directory, unet, controlnet, step, *, final=False):
    """Publish a matched control branch and target adapters at one update.

    The frozen U-Net backbone is stored once in evaluation_assets/unet.
    These are inference snapshots, not resumable optimizer checkpoints.
    """
    directory = Path(directory)
    destination = directory / "evaluation_queue/jobs" / f"gs_{step:06d}"
    completed = directory / "evaluation_results" / f"gs_{step}.json"
    if completed.is_file():
        result = json.loads(completed.read_text())
        if result.get("state") != "completed" or result.get("step") != step or result.get("final") != bool(final):
            raise ValueError("An incompatible completed evaluation already exists.")
        return destination
    if destination.exists():
        job = json.loads((destination / "job.json").read_text())
        if job["step"] != step or job["final"] != final:
            raise ValueError("An incompatible evaluation snapshot already exists.")
        return destination
    temporary = destination.with_name("." + destination.name + ".incomplete")
    temporary.mkdir(parents=True, exist_ok=False)
    controlnet.save_pretrained(temporary / "controlnet", safe_serialization=False)
    adapters = {key: value.detach().cpu().clone() for key, value in unet.metadata_embedding.state_dict().items()}
    torch.save(adapters, temporary / "target_metadata.bin")
    files = {path.relative_to(temporary).as_posix(): path.stat().st_size
             for path in temporary.rglob("*") if path.is_file()}
    write_json(temporary / "job.json", {
        "step": step, "final": bool(final), "inference_only": bool(final),
        "runtime_version": RUNTIME_VERSION, "evaluation_only": True, "files": files,
    })
    temporary.rename(destination)
    return destination


def validate_snapshot(path):
    path = Path(path)
    job = json.loads((path / "job.json").read_text())
    required = {"controlnet/config.json", "controlnet/diffusion_pytorch_model.bin", "target_metadata.bin"}
    if not required.issubset(job["files"]) or not job.get("evaluation_only"):
        raise ValueError(f"Incomplete evaluation snapshot: {path}")
    for name, size in job["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid evaluation snapshot path")
        if not (path / relative).is_file() or (path / relative).stat().st_size != size:
            raise ValueError(f"Incomplete evaluation snapshot file: {path / relative}")
    return job


class AsyncEvaluationQueue:
    def __init__(self, directory, args, unet, run_config, dataset_split):
        self.directory = Path(directory).resolve()
        self.process = None
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "evaluation_queue").mkdir(exist_ok=True)
        status_file = self.directory / "evaluation_status.json"
        if status_file.exists():
            previous = json.loads(status_file.read_text())
            if previous.get("state") in {"starting", "running", "evaluating"}:
                try:
                    os.kill(previous["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError("An evaluation worker is already active for this run.")
            # A failed worker's status must not stop the first update of a
            # resumed run while its replacement is still importing modules.
            status_file.unlink()
        source = Path(__file__).resolve().parent.parent
        frozen = self.directory / "evaluation_source"
        if not frozen.exists():
            temporary = frozen.with_name(".evaluation_source.incomplete")
            temporary.mkdir()
            for folder in ("orgacast", "vendor/diffusers/src"):
                shutil.copytree(source / folder, temporary / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            shutil.copy2(source / "train_controlnet.py", temporary / "train_controlnet.py")
            (temporary / "tools").mkdir()
            shutil.copy2(source / "tools/evaluate_queue.py", temporary / "tools/evaluate_queue.py")
            temporary.rename(frozen)
        # Reject stale copied code on resume instead of silently mixing versions.
        import hashlib
        for name, digest in run_config["source_sha256"].items():
            if hashlib.sha256((frozen / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Evaluation source differs from training: {name}")
        base = self.directory / "evaluation_assets/unet"
        if not base.exists():
            temporary = base.with_name(".unet.incomplete")
            temporary.mkdir(parents=True)
            unet.save_pretrained(temporary, safe_serialization=False)
            temporary.rename(base)
        # Only explicitly selected arguments are persisted; no credentials.
        evaluation_args = dict(run_config)
        evaluation_args.update(
            output_dir=str(self.directory), run_uncertainty_analysis=args.run_uncertainty_analysis,
            mc_samples=args.mc_samples, uncertainty_output_dir=args.uncertainty_output_dir,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path, revision=args.revision,
        )
        write_json(self.directory / "evaluation_queue/config.json", {
            "args": evaluation_args, "split": dataset_split, "working_directory": str(Path.cwd()),
            "training_pid": os.getpid(), "evaluation_gpu": args.evaluation_gpu,
        })
        self.close("running")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = args.evaluation_gpu
        environment["PYTHONPATH"] = os.pathsep.join([str(frozen / "vendor/diffusers/src"), str(frozen)])
        environment["PYTHONUNBUFFERED"] = "1"
        environment["OMP_NUM_THREADS"] = "4"
        environment["MKL_NUM_THREADS"] = "4"
        with (self.directory / "evaluation_worker.log").open("ab", buffering=0) as log:
            self.process = subprocess.Popen(
                [sys.executable, str(frozen / "tools/evaluate_queue.py"), "--output_dir", str(self.directory)],
                cwd=Path.cwd(), env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        write_json(self.directory / "evaluation_worker_process.json", {"pid": self.process.pid, "gpu": args.evaluation_gpu})

    def submit(self, unet, controlnet, step, *, final=False):
        return publish_evaluation_snapshot(self.directory, unet, controlnet, step, final=final)

    def error(self):
        path = self.directory / "evaluation_status.json"
        if path.exists():
            status = json.loads(path.read_text())
            if status.get("state") == "failed":
                return status.get("error", "Evaluation worker failed; see evaluation_worker.log")
        if self.process is not None and self.process.poll() is not None:
            return f"Evaluation worker exited unexpectedly with code {self.process.returncode}"
        return None

    def close(self, state):
        path = self.directory / "evaluation_queue/producer.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"state": state, "pid": os.getpid()})
