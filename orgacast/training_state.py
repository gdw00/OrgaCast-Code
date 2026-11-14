"""Joint model checkpoints and deterministic progress for cell training."""

from contextlib import contextmanager, nullcontext
import json
import hashlib
from pathlib import Path
import re
import signal

import torch
from accelerate.utils import broadcast_object_list
from torch.utils.data import Subset

from .masking import mask_configuration
from .image_losses import area_loss_configuration
from .paper_training import METHOD_VERSION, METADATA_VERSION
from .text_conditioning import text_conditioning_configuration


FORMAT_VERSION = 2
STATE_FILE = "training_state.json"
SPLIT_FILE = "dataset_split.json"
COMPLETE_FILE = "checkpoint_complete.json"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def cell_dataset_split(dataset, seed, saved=None):
    """Keep the 80/20 policy and fixed validation sample; persist sample IDs."""
    samples = list(dataset.samples)
    if len(samples) != len(set(samples)):
        raise ValueError("Cell sample IDs must be unique to restore a split.")
    if saved is None:
        fixed = "ST_ctrl_XY00"
        candidates = [key for key in samples if key != fixed]
        order = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(seed)).tolist()
        ordered = [candidates[index] for index in order]
        train_count = int(0.8 * len(ordered))
        saved = {"train": ordered[:train_count], "validation": ordered[train_count:]}
        if fixed in samples:
            saved["validation"].append(fixed)
    train_keys, validation_keys = saved["train"], saved["validation"]
    all_keys = train_keys + validation_keys
    if len(all_keys) != len(set(all_keys)) or set(all_keys) != set(samples):
        raise ValueError("Dataset sample IDs differ from the checkpoint split; start a new run.")
    if not train_keys or not validation_keys:
        raise ValueError("Both training and validation splits must contain cell sequences.")
    index = {key: position for position, key in enumerate(samples)}
    return (
        Subset(dataset, [index[key] for key in train_keys]),
        Subset(dataset, [index[key] for key in validation_keys]),
        saved,
    )


def read_training_checkpoint(path):
    """Reject legacy weights and partially written checkpoints before training."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Training checkpoint does not exist: {path}")
    marker = path / COMPLETE_FILE
    if not marker.is_file():
        raise ValueError(
            f"{path} is not a complete joint-training checkpoint. For legacy ControlNet weights, "
            "use --controlnet_model_name_or_path=<checkpoint>/controlnet to start a new stage."
        )
    manifest = json.loads(marker.read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format in {path}. The paper-aligned method starts a new run; "
            "use the archived pre-alignment source to resume a format-1 historical run."
        )
    files = manifest.get("files", {})
    required = {
        STATE_FILE, SPLIT_FILE, "unet/config.json", "unet/diffusion_pytorch_model.bin",
        "controlnet/config.json", "controlnet/diffusion_pytorch_model.bin",
        "optimizer.bin", "scheduler.bin", "random_states_0.pkl",
    }
    if not required.issubset(files):
        raise ValueError(f"Checkpoint manifest is incomplete: {path}")
    for relative, size in files.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Invalid path in checkpoint manifest")
        file = path / rel
        if not file.is_file() or file.stat().st_size != size:
            raise ValueError(f"Missing or incomplete checkpoint file: {file}")
    state = json.loads((path / STATE_FILE).read_text())
    split = json.loads((path / SPLIT_FILE).read_text())
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError("Incompatible training state version")
    for name in ("global_step", "epoch", "next_batch"):
        if type(state.get(name)) is not int or state[name] < 0:
            raise ValueError(f"Invalid training progress field: {name}")
    if not isinstance(state.get("run_config"), dict) or not isinstance(state.get("initialization"), dict):
        raise ValueError("Checkpoint is missing its training configuration or initialization provenance")
    for rank in range(state["run_config"]["num_processes"]):
        if f"random_states_{rank}.pkl" not in files:
            raise ValueError(f"Checkpoint is missing random state for process {rank}")
    if state["run_config"]["mixed_precision"] == "fp16" and "scaler.pt" not in files:
        raise ValueError("Checkpoint is missing its mixed-precision scaler")
    if not isinstance(split.get("train"), list) or not isinstance(split.get("validation"), list):
        raise ValueError("Checkpoint has no valid dataset split")
    return state, split


def resolve_resume_checkpoint(value, output_dir):
    if value is None:
        return None
    if value == "latest":
        candidates = []
        for path in Path(output_dir).glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if match and (path / COMPLETE_FILE).is_file():
                candidates.append((int(match.group(1)), path))
        if not candidates:
            raise FileNotFoundError(f"No complete OrgaCast training checkpoint in {output_dir}")
        path = max(candidates, key=lambda item: item[0])[1]
    else:
        path = Path(value)
    read_training_checkpoint(path)
    return path


def training_run_config(args, accelerator):
    # Explicit allowlist: credentials and machine-specific output locations are
    # not checkpoint configuration. Initial model paths are recorded separately.
    fields = (
        "seed", "resolution", "cond_resolution", "num_metadata", "num_cond", "num_channels",
        "temporal", "temporal_attn", "multi", "train_batch_size", "gradient_accumulation_steps",
        "dataloader_num_workers", "max_train_steps", "learning_rate", "scale_lr",
        "adam_beta1", "adam_beta2", "adam_weight_decay", "adam_epsilon", "use_8bit_adam",
        "lr_scheduler", "lr_warmup_steps", "lr_num_cycles", "lr_power", "max_grad_norm",
        "gradient_checkpointing", "set_grads_to_none", "allow_tf32",
        "enable_xformers_memory_efficient_attention", "drop_cond", "drop_temporal_cond",
        "controlnet_scales", "morph_loss_lambda", "morph_temperature", "validation_steps", "test_second_day_only",
        "pretrained_model_name_or_path", "tokenizer_name", "revision", "train_data_dir",
        "input_days", "conditioning_ablation", "metadata_dropout", "temporal_dropout",
        "temporal_kernel_size", "image_mse_weight", "lpips_loss_weight", "diffusion_loss_weight",
        "metadata_learning_rate", "num_inference_steps", "vae_path",
        "cache_preprocessed_data", "cache_target_masks", "evaluation_mode", "evaluation_gpu",
        "validate_initialization", "skip_final_pipeline_export",
        "text_encoding_mode", "text_max_chunks",
    )
    result = {name: getattr(args, name) for name in fields}
    result.update(
        num_processes=accelerator.num_processes,
        mixed_precision=accelerator.mixed_precision,
        unet_backbone_trainable=False,
        target_metadata_trainable=True,
        method_version=METHOD_VERSION,
        preprocessing="history_minmax_channel_extract_v1",
        metadata_semantics=METADATA_VERSION,
        hard_otsu_status="hard_forward_sigmoid_surrogate_backward",
        foreground_mask=mask_configuration(),
        foreground_mask_target="actual_preprocessed_target",
        foreground_area_loss=area_loss_configuration(args.morph_temperature),
        inference_data_dir=args.inference_data_dir,
        data_order="saved_split_sequential",
        inference_only_schedule="final",
        runtime_version="cached_data_async_evaluation_v1",
        text_conditioning=text_conditioning_configuration(args),
    )
    source_root = Path(__file__).resolve().parent.parent
    source_paths = [source_root / "train_controlnet.py", *sorted((source_root / "orgacast").glob("*.py")),
                    source_root / "tools/evaluate_queue.py"]
    result["source_sha256"] = {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths if path.is_file()
    }
    return result


def check_resume_config(saved, current):
    differences = [name for name in sorted(set(saved) | set(current)) if saved.get(name) != current.get(name)]
    if differences:
        detail = ", ".join(f"{name}: saved={saved.get(name)!r}, requested={current.get(name)!r}" for name in differences)
        raise ValueError("Resume configuration differs from the saved run (start a new stage for changes): " + detail)


def assert_optimizer_binding(optimizer, *models):
    expected = {id(parameter) for model in models for parameter in model.parameters() if parameter.requires_grad}
    actual = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise RuntimeError("Optimizer parameters do not exactly match the current trainable U-Net and ControlNet.")


@contextmanager
def accumulate_joint_models(accelerator, controlnet, unet):
    """Accelerate 0.21 accepts one model in accumulate; advance its counter once."""
    with accelerator.accumulate(controlnet):
        context = nullcontext() if accelerator.sync_gradients else accelerator.no_sync(unet)
        with context:
            yield


def register_joint_model_hooks(accelerator, unet, controlnet):
    named = {"unet": accelerator.unwrap_model(unet), "controlnet": accelerator.unwrap_model(controlnet)}
    names_by_id = {id(model): name for name, model in named.items()}

    def model_name(model):
        model = accelerator.unwrap_model(model)
        if id(model) not in names_by_id:
            raise RuntimeError("Unexpected model registered for joint checkpointing")
        return names_by_id[id(model)], model

    def save_hook(models, weights, directory):
        for index in range(len(models) - 1, -1, -1):
            name, model = model_name(models[index])
            if accelerator.is_main_process:
                # MultiControlNetModel already creates controlnet/ subfolders.
                destination = Path(directory) if hasattr(model, "nets") else Path(directory) / name
                model.save_pretrained(destination, safe_serialization=False)
            weights.pop(index)

    def load_hook(models, directory):
        while models:
            name, model = model_name(models.pop())
            # Copy tensors into the existing Parameter objects; the optimizer
            # and distributed wrappers must keep referencing these objects.
            if hasattr(model, "nets"):
                loaded = type(model).from_pretrained(str(directory), local_files_only=True)
                model.load_state_dict(loaded.state_dict(), strict=True)
                del loaded
            else:
                state = torch.load(Path(directory) / name / "diffusion_pytorch_model.bin", map_location="cpu")
                model.load_state_dict(state, strict=True)
                del state

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)


def _main_file_operation(accelerator, operation):
    error = [None]
    if accelerator.is_main_process:
        try:
            operation()
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if accelerator.num_processes > 1:
        broadcast_object_list(error)
    if error[0] is not None:
        raise RuntimeError(error[0])


def save_training_checkpoint(accelerator, path, state, split):
    """Publish a checkpoint only after every rank has saved its complete state."""
    path = Path(path)
    temporary = path.with_name("." + path.name + ".incomplete")

    def prepare_directory():
        if path.exists() or temporary.exists():
            raise FileExistsError(f"Refusing to overwrite {path} or {temporary}")
        temporary.mkdir(parents=True)

    _main_file_operation(accelerator, prepare_directory)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(temporary))
    accelerator.wait_for_everyone()

    def publish():
        write_json(temporary / STATE_FILE, dict(state, format_version=FORMAT_VERSION))
        write_json(temporary / SPLIT_FILE, split)
        files = {p.relative_to(temporary).as_posix(): p.stat().st_size for p in temporary.rglob("*") if p.is_file()}
        write_json(temporary / COMPLETE_FILE, {"format_version": FORMAT_VERSION, "files": files})
        read_training_checkpoint(temporary)
        temporary.rename(path)

    _main_file_operation(accelerator, publish)
    accelerator.wait_for_everyone()


class StopAtUpdateBoundary:
    """SIGINT/SIGTERM request a checkpoint after the next completed update."""

    def __init__(self):
        self.requested = False
        self.previous = {}

    def __enter__(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.signal(signum, self._request)
        return self

    def _request(self, signum, frame):
        self.requested = True

    def across_processes(self, accelerator):
        if accelerator.num_processes > 1:
            flag = torch.tensor(int(self.requested), device=accelerator.device)
            self.requested = bool(accelerator.reduce(flag, reduction="sum").item())
        return self.requested

    def __exit__(self, *exception):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)
