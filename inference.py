"""Forecast or evaluate cells using paper-core component checkpoints."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HOME', './hf_cache')

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler
from tqdm.auto import tqdm

from orgacast import OrgaCastUNet, ControlNetModel3D
from orgacast.cell_data import MultiConditionTimeSeriesDataset, InferenceOnlyDataset, collate_cells, PREPROCESS_VERSION
from orgacast.forecast import forecast_batch
from orgacast.evaluation import image_metrics, save_metric_records
from orgacast.paper_training import METADATA_VERSION, METHOD_VERSION
from orgacast.masking import foreground_masks, mask_configuration
from orgacast.training_state import read_training_checkpoint, write_json
from orgacast.text_conditioning import caption_audit, text_conditioning_configuration


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='Complete format-2 checkpoint directory.')
    parser.add_argument('--pretrained_model_name_or_path')
    parser.add_argument('--vae_path')
    parser.add_argument('--unet_path')
    parser.add_argument('--controlnet_path')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='./outputs/orgacast_predictions')
    parser.add_argument('--with_targets', action='store_true', help='Load Day 15 and compute evaluation metrics.')
    parser.add_argument('--subset', choices=['all', 'train', 'validation'])
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--num_metadata', type=int)
    parser.add_argument('--num_cond', type=int)
    parser.add_argument('--input_days', nargs='+', choices=['day7', 'day9', 'day11'])
    parser.add_argument('--conditioning_ablation', choices=['none', 'text', 'metadata', 'images'])
    parser.add_argument('--text_encoding_mode', choices=['truncate', 'chunked'])
    parser.add_argument('--text_max_chunks', type=int)
    parser.add_argument('--controlnet_scales', type=float, nargs='+')
    parser.add_argument('--inference_batch_size', type=int, default=4)
    parser.add_argument('--num_inference_steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--online', action='store_true', help='Permit model downloads; otherwise use existing local assets.')
    # Retained command-line spellings; this entry uses temporal models only.
    parser.add_argument('--temporal', action='store_true', default=True)
    parser.add_argument('--temporal_attn', action='store_true', default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    saved, split = ({}, None)
    if args.checkpoint:
        saved, split = read_training_checkpoint(args.checkpoint)
        args.unet_path = args.unet_path or str(Path(args.checkpoint) / 'unet')
        args.controlnet_path = args.controlnet_path or str(Path(args.checkpoint) / 'controlnet')
    config = saved.get('run_config', {})
    initialization = saved.get('initialization', {})
    args.pretrained_model_name_or_path = args.pretrained_model_name_or_path or config.get('pretrained_model_name_or_path')
    args.vae_path = args.vae_path or initialization.get('vae')
    for name in ['pretrained_model_name_or_path', 'vae_path', 'unet_path', 'controlnet_path']:
        if not getattr(args, name):
            raise ValueError(f'Provide --{name}, or a checkpoint that records it.')
    for name, default in [('resolution', 256), ('num_metadata', 4), ('input_days', ['day7','day9','day11']),
                          ('conditioning_ablation', 'none'), ('num_inference_steps', 50), ('seed', 42),
                          ('text_encoding_mode', 'truncate'), ('text_max_chunks', 3)]:
        if getattr(args, name) is None:
            setattr(args, name, config.get(name, default))
    if args.controlnet_scales is None:
        args.controlnet_scales = config.get('controlnet_scales')
    args.num_cond = len(args.input_days) if args.num_cond is None else args.num_cond
    if args.num_metadata != 4 or args.num_cond != len(args.input_days):
        raise ValueError('Use four cell metadata fields and one condition per observed day.')
    if args.inference_batch_size < 1 or args.num_inference_steps < 1 or args.resolution < 16 or args.resolution % 8:
        raise ValueError('Invalid batch size, sampling steps or resolution.')
    if not args.input_days or len(set(args.input_days)) != len(args.input_days) or args.input_days != sorted(args.input_days, key=lambda day: int(day[3:])):
        raise ValueError('Input days must be unique and chronological.')
    if args.controlnet_scales is not None and len(args.controlnet_scales) != args.num_cond:
        raise ValueError('Provide one control scale for each input day.')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'prediction_config.json').exists():
        raise ValueError('Prediction output already exists; select a new output directory.')
    local = not args.online
    tokenizer_source = config.get('tokenizer_name') or args.pretrained_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source,
        subfolder=None if config.get('tokenizer_name') else 'tokenizer',
        revision=config.get('revision'), use_fast=False, local_files_only=local)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder', local_files_only=local)
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=local)
    unet = OrgaCastUNet.from_pretrained(args.unet_path, local_files_only=local)
    controlnet = ControlNetModel3D.from_pretrained(args.controlnet_path, local_files_only=local)
    if getattr(unet.config, 'metadata_semantics', None) != METADATA_VERSION or controlnet.config.temporal_fusion_mode != 'parallel' or controlnet.config.metadata_conditioning != 'independent':
        raise ValueError('These components are not the paper-core architecture; use the archived entry for historical weights.')
    for model in [unet, controlnet, vae, text_encoder]:
        model.requires_grad_(False).eval().to(args.device, dtype=torch.float32)
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler', local_files_only=local)
    dataset_class = MultiConditionTimeSeriesDataset if args.with_targets else InferenceOnlyDataset
    dataset = dataset_class(args.data_dir, tokenizer=tokenizer, args=args)
    write_json(output / 'text_conditioning_audit.json', caption_audit(tokenizer, args, dataset.caption_texts))
    subset = args.subset or ('validation' if args.with_targets and split is not None else 'all')
    selected_ids = list(dataset.samples)
    if subset != 'all':
        if split is None:
            raise ValueError('--subset requires a checkpoint with a saved dataset split.')
        selected_ids = split[subset]
        index = {key: i for i, key in enumerate(dataset.samples)}
        missing = set(selected_ids) - set(index)
        if missing:
            raise ValueError(f'Data directory is missing saved samples: {sorted(missing)}')
        dataset = Subset(dataset, [index[key] for key in selected_ids])
    if not len(dataset):
        raise ValueError('No usable cell sequences found.')
    loader = DataLoader(dataset, batch_size=args.inference_batch_size, shuffle=False, collate_fn=collate_cells)
    perceptual_model = None
    if args.with_targets:
        import lpips
        perceptual_model = lpips.LPIPS(net='vgg').requires_grad_(False).eval().to(args.device)
    write_json(output / 'prediction_config.json', dict(vars(args), method_version=METHOD_VERSION,
               checkpoint_method_version=config.get('method_version'),
               preprocessing=PREPROCESS_VERSION, sample_ids=selected_ids, subset_used=subset,
               foreground_mask=mask_configuration(), text_conditioning=text_conditioning_configuration(args)))
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    records = []
    for batch in tqdm(loader, desc='Forecasting Day 15'):
        predicted = forecast_batch(unet, controlnet, vae, text_encoder, scheduler, batch, args, generator)
        predicted_masks = foreground_masks(predicted)
        if args.with_targets:
            targets = (batch['pixel_values'].to(predicted.device) / 2 + 0.5).clamp(0, 1)
            target_masks = foreground_masks(targets)
        for index, key in enumerate(batch['sample_keys']):
            save_image(predicted[index][[1,0,2]], output / f'{key}_predicted_RGB_swapped.png')
            save_image(predicted_masks[index], output / f'{key}_predicted_mask.png')
            save_image(predicted[index].amax(dim=0, keepdim=True), output / f'{key}_predicted_MAX_OVERLAY.png')
            # Preserve the actual model channels independently of display colors.
            np.save(output / f'{key}_prediction.npy', predicted[index].cpu().numpy())
            conditions = (batch['conditioning_pixel_values'][index] / 2 + 0.5).clamp(0,1)
            save_image(conditions[:,[1,0,2]], output / f'{key}_conditions_grid.png', nrow=args.num_cond)
            if args.with_targets:
                target = targets[index]
                records.append(dict(sample_key=key, **image_metrics(predicted[index], target, perceptual_model)))
                save_image(target[[1,0,2]], output / f'{key}_target.png')
                save_image(target_masks[index], output / f'{key}_target_mask.png')
    if records:
        print(json.dumps(save_metric_records(output / 'metrics.json', records), indent=2))
    print(f'Saved {len(dataset)} forecasts to {output}')


if __name__ == '__main__':
    main()
