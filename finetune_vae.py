"""Fine-tune a VAE on the training split of organoid images."""
import argparse
import json
from types import SimpleNamespace
import logging
import os
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from transformers import AutoTokenizer


from orgacast.cell_data import MultiConditionTimeSeriesDataset, collate_cells, PREPROCESS_VERSION
from orgacast.training_state import cell_dataset_split, write_json

logger = get_logger(__name__)
def parse_vae_args():
    parser = argparse.ArgumentParser(description="Fine-tuning a VAE for a specific domain.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1", required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--train_data_dir", type=str, required=True, help="A folder containing the training data.")
    parser.add_argument("--output_dir", type=str, default="./outputs/finetuned-vae", help="The output directory where the model checkpoints will be written.")


    parser.add_argument("--resolution", type=int, default=256, help="The resolution for input images.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_steps", type=int, default=10000, help="Total number of training steps to perform.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")


    parser.add_argument("--lr_scheduler", type=str, default="constant", help='The scheduler type to use. Choose between ["linear", "cosine", "constant", ...]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--checkpointing_steps", type=int, default=1000, help="Save a checkpoint of the training state every X updates.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help='The integration to report the results and logs to.')
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--split_file", help="Saved sample split shared with the forecasting experiment.")
    args = parser.parse_args()
    return args

def main():
    args = parse_vae_args()


    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)


    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")


    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    data_args = SimpleNamespace(num_metadata=4, resolution=args.resolution,
                                input_days=["day7", "day9", "day11"], conditioning_ablation="none")
    full_dataset = MultiConditionTimeSeriesDataset(
        root_dir=args.train_data_dir, transform=None, tokenizer=tokenizer, args=data_args,
    )
    saved_split = json.loads(Path(args.split_file).read_text()) if args.split_file else None
    train_dataset, _, split = cell_dataset_split(full_dataset, args.seed, saved=saved_split)
    if accelerator.is_main_process:
        write_json(Path(args.output_dir) / "dataset_split.json", split)
        write_json(Path(args.output_dir) / "data_config.json", {
            "train_data_dir": args.train_data_dir, "preprocessing": PREPROCESS_VERSION,
            "resolution": args.resolution, "seed": args.seed,
        })

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_cells,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )


    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )


    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("vae-finetune", config=vars(args))


    logger.info("***** Running VAE fine-tuning *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    vae.train()
    data_iter = iter(train_dataloader)
    while global_step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:

            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
        reconstructed_pixels = vae(pixel_values).sample
        loss = F.mse_loss(reconstructed_pixels, pixel_values, reduction="mean")

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"loss": loss.item()}, step=global_step)

        if accelerator.is_main_process:
            if global_step % args.checkpointing_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                unwrapped_vae = accelerator.unwrap_model(vae)
                unwrapped_vae.save_pretrained(save_path)
                logger.info(f"Saved VAE checkpoint to {save_path}")


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_vae = accelerator.unwrap_model(vae)
        unwrapped_vae.save_pretrained(args.output_dir)
        logger.info(f"Saved final VAE model to {args.output_dir}")

    accelerator.end_training()

if __name__ == "__main__":
    main()
