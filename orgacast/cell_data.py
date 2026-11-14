"""Shared cell sequence loading for training, validation and unlabeled inference.

Statistics are fitted only to observed days. Raw four-field metadata follows the
paper; seven-field scaling remains an explicit adapter for historical weights.
"""
import os
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .data_util import metadata_normalize
from .captions import cell_caption
from .text_conditioning import tokenize_caption

PREPROCESS_VERSION = "history_minmax_channel_extract_v1"
CONDITION_NAMES = ("DY_ctrl", "DY_DAPT", "ST_ctrl", "ST_DAPT", "2DY_ctrl", "2DY_DAPT", "2ST_ctrl", "2ST_DAPT", "DY_GF_Pillars", "DY_GF_V_Pillars", "ST_GF_Pillars")
# Stored tensor order is CH1, CH2, CH3; RGB display order is CH2, CH1, CH3.
RGB_COMPONENT = {"CH1": 1, "CH2": 0, "CH3": 2}


def canonical_condition(value):
    return next(name for name in CONDITION_NAMES if name.casefold() == value.casefold())


def read_fluorescence_image(path, channel):
    with Image.open(path) as image:
        values = np.asarray(image)
        if values.ndim == 3:
            # Input TIFFs may be fluorescence rendered into one RGB component.
            # Do not apply luminance weights to the biological intensity channel.
            values = values[..., RGB_COMPONENT[channel]]
        if values.ndim != 2:
            raise ValueError(f"Expected a single fluorescence plane: {path}")
        return Image.fromarray(values.astype(np.float32, copy=True))


def collate_cells(examples):
    if not examples or any(example is None for example in examples):
        raise ValueError("A cell sequence could not be read; fix the data before training or inference.")
    batch = {key: torch.stack([example[key] for example in examples])
             for key in examples[0] if key != "sample_key"}
    batch["sample_keys"] = [example["sample_key"] for example in examples]
    return batch


def history_transforms(images, input_days, channels, resolution):
    transforms_by_channel = {}
    for channel in channels:
        lower = min(float(np.asarray(images[day][channel]).min()) for day in input_days)
        upper = max(float(np.asarray(images[day][channel]).max()) for day in input_days)
        upper = max(upper, lower + 1e-6)
        transforms_by_channel[channel] = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(resolution),
            ApplyLinearContrastEnhancement(global_min=lower, global_max=upper),
            transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
        ])
    return transforms_by_channel


class ApplyLinearContrastEnhancement:
    """Stretch grayscale intensity or apply linear contrast around 128."""

    def __init__(self, contrast_factor=1.5, method='histogram_stretch', global_min=None, global_max=None):
        """Use supplied intensity limits, or fit them to each image."""
        self.contrast_factor = contrast_factor
        self.method = method
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, img):
        """Return an 8-bit grayscale PIL image."""
        img_np = np.array(img, dtype=np.float32)
        if self.method == 'histogram_stretch':
            if self.global_min is not None and self.global_max is not None:
                img_min = self.global_min
                img_max = self.global_max
            else:
                img_min = img_np.min()
                img_max = img_np.max()
            if img_max > img_min:
                enhanced_np = (img_np - img_min) / (img_max - img_min) * 255
            else:
                enhanced_np = img_np
        else:
            enhanced_np = (img_np - 128) * self.contrast_factor + 128
        enhanced_np = np.clip(enhanced_np, 0, 255).astype(np.uint8)
        return Image.fromarray(enhanced_np)

class MultiConditionTimeSeriesDataset(Dataset):

    def __init__(self, root_dir, transform=None, tokenizer=None, args=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.args = args
        self.input_days = list(getattr(args, 'input_days', ['day7', 'day9', 'day11']))
        self.target_day = 'day15'
        self.channels = ['CH1', 'CH2', 'CH3']
        self.condition_day_folders = {}
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        folder_pattern = re.compile('.*_(DY_ctrl|DY_DAPT|ST_ctrl|ST_DAPT|2DY_ctrl|2DY_DAPT|2ST_ctrl|2ST_DAPT|DY_GF_Pillars|DY_GF_V_Pillars|ST_GF_Pillars)_(day\\d+)_consistent_settings', re.IGNORECASE)
        for sd in subdirs:
            match = folder_pattern.match(sd)
            if match:
                condition = canonical_condition(match.group(1))
                day = match.group(2).lower()
                path = os.path.join(root_dir, sd)
                key = (condition, day)
                if key in self.condition_day_folders:
                    raise ValueError(f'Duplicate condition/day folders: {key}')
                self.condition_day_folders[key] = path
        self.sample_pattern = re.compile('^Image_(XY\\d+)_CH\\d+\\.tif$', re.IGNORECASE)
        self.condition_to_id = {'DY_ctrl': 0, 'DY_DAPT': 1, 'ST_ctrl': 2, 'ST_DAPT': 3, '2DY_ctrl': 0, '2DY_DAPT': 1, '2ST_ctrl': 2, '2ST_DAPT': 3, 'DY_GF_Pillars': 0, 'DY_GF_V_Pillars': 0, 'ST_GF_Pillars': 2}
        sample_set = set()
        possible_conditions = []
        for cond, d in self.condition_day_folders.keys():
            if d == self.input_days[0]:
                possible_conditions.append(cond)
        for cond in possible_conditions:
            day7_folder = self.condition_day_folders.get((cond, self.input_days[0]), None)
            if day7_folder is None:
                continue
            for fname in os.listdir(day7_folder):
                match = self.sample_pattern.match(fname)
                if match:
                    xy = match.group(1)
                    sample_key = f'{cond}_{xy}'
                    sample_set.add(sample_key)
        valid_samples = []
        for sk in sample_set:
            parts = sk.split('_')
            if len(parts) >= 2:
                xy = parts[-1]
                cond = '_'.join(parts[:-1])
                if self.check_valid(cond, xy):
                    valid_samples.append(sk)
            else:
                print(f'Skipping invalid sample key format: {sk}')
        self.samples = sorted(valid_samples)
        self.caption_texts = {
            condition: cell_caption(condition, self.target_day)
            if getattr(self.args, "conditioning_ablation", "none") != "text" else ""
            for condition in sorted({key.rsplit("_", 1)[0] for key in self.samples})
        }
        print(f'Found {len(self.samples)} valid (condition+XY) samples after filtering.')

    def check_valid(self, cond, xy):
        days_needed = self.input_days + [self.target_day]
        for day in days_needed:
            folder = self.condition_day_folders.get((cond, day), None)
            if folder is None:
                return False
            for ch in self.channels:
                fname = f'Image_{xy}_{ch}.tif'
                fpath = os.path.join(folder, fname)
                if not os.path.exists(fpath):
                    return False
        return True

    def load_image(self, folder, xy, ch):
        fname = f'Image_{xy}_{ch}.tif'
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            print(f'Warning: File not found {path} during loading')
            return None
        try:
            img = read_fluorescence_image(path, ch)
            return img
        except Exception as e:
            print(f'Error loading image {path}: {e}')
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            print(f'Error: Index {idx} out of bounds')
            return None
        sample_key = self.samples[idx]
        parts = sample_key.split('_')
        if len(parts) < 2:
            print(f'Sample key format error: {sample_key}')
            return None
        xy = parts[-1]
        condition = '_'.join(parts[:-1])
        pil_images_by_day = {}
        days_to_scan = self.input_days + [self.target_day]
        for day in days_to_scan:
            folder = self.condition_day_folders.get((condition, day), None)
            if folder is None:
                print(f'Missing folder for {sample_key} on {day}. Skipping sample.')
                return None
            pil_images_by_day[day] = {}
            for ch in self.channels:
                img_pil = self.load_image(folder, xy, ch)
                if img_pil is None:
                    print(f'Missing file {ch} on {day} for {sample_key}. Skipping sample.')
                    return None
                pil_images_by_day[day][ch] = img_pil
        channel_transforms = history_transforms(pil_images_by_day, self.input_days, self.channels, self.args.resolution)
        input_images_list = []
        for day in self.input_days:
            transformed_day_tensors = []
            for ch in self.channels:
                img_pil = pil_images_by_day[day][ch]
                tensor = channel_transforms[ch](img_pil)
                transformed_day_tensors.append(tensor)
            day_tensor = torch.cat(transformed_day_tensors, dim=0)
            input_images_list.append(day_tensor)
        conditioning_pixel_values = torch.stack(input_images_list, dim=0)
        transformed_target_tensors = []
        for ch in self.channels:
            img_pil = pil_images_by_day[self.target_day][ch]
            tensor = channel_transforms[ch](img_pil)
            transformed_target_tensors.append(tensor)
        pixel_values = torch.cat(transformed_target_tensors, dim=0)
        text_prompt = self.caption_texts[condition]
        if getattr(self.args, 'conditioning_ablation', 'none') == 'text':
            text_prompt = ''
        input_ids = tokenize_caption(self.tokenizer, text_prompt, self.args)
        num_metadata = self.args.num_metadata
        target_metadata_values = torch.zeros(num_metadata, dtype=torch.float32)
        if num_metadata > 0:
            target_metadata_values[0] = float(self.condition_to_id.get(condition, -1))
        if num_metadata > 1:
            target_metadata_values[1] = 1.0 if 'DY' in condition else 0.0
        if num_metadata > 2:
            target_metadata_values[2] = 1.0 if 'DAPT' in condition else 0.0
        if num_metadata > 3:
            target_metadata_values[3] = float(self.target_day.replace('day', ''))
        metadata = metadata_normalize(target_metadata_values)
        conditioning_metadata_list = []
        for day_str in self.input_days:
            cond_meta_values = torch.zeros(num_metadata, dtype=torch.float32)
            if num_metadata > 0:
                cond_meta_values[0] = float(self.condition_to_id.get(condition, -1))
            if num_metadata > 1:
                cond_meta_values[1] = 1.0 if 'DY' in condition else 0.0
            if num_metadata > 2:
                cond_meta_values[2] = 1.0 if 'DAPT' in condition else 0.0
            if num_metadata > 3:
                cond_meta_values[3] = float(day_str.replace('day', ''))
            conditioning_metadata_list.append(metadata_normalize(cond_meta_values))
        conditioning_metadata = torch.stack(conditioning_metadata_list, dim=-1)
        if getattr(self.args, 'conditioning_ablation', 'none') == 'metadata':
            metadata = torch.zeros_like(metadata)
            conditioning_metadata = torch.zeros_like(conditioning_metadata)
        return {'pixel_values': pixel_values, 'conditioning_pixel_values': conditioning_pixel_values, 'input_ids': input_ids, 'metadata': metadata, 'conditioning_metadata': conditioning_metadata, 'sample_key': sample_key}

class InferenceOnlyDataset(Dataset):
    """Load observed days without requiring a target-day image."""

    def __init__(self, root_dir, transform=None, tokenizer=None, args=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.args = args
        self.input_days = list(getattr(args, 'input_days', ['day7', 'day9', 'day11']))
        self.target_day = 'day15'
        self.channels = ['CH1', 'CH2', 'CH3']
        self.condition_day_folders = {}
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        folder_pattern = re.compile('.*_(DY_ctrl|DY_DAPT|ST_ctrl|ST_DAPT|2DY_ctrl|2DY_DAPT|2ST_ctrl|2ST_DAPT|DY_GF_Pillars|DY_GF_V_Pillars|ST_GF_Pillars)_(day\\d+)_consistent_settings', re.IGNORECASE)
        for sd in subdirs:
            match = folder_pattern.match(sd)
            if match:
                condition = canonical_condition(match.group(1))
                day = match.group(2).lower()
                path = os.path.join(root_dir, sd)
                key = (condition, day)
                if key in self.condition_day_folders:
                    raise ValueError(f'Duplicate condition/day folders: {key}')
                self.condition_day_folders[key] = path
        self.sample_pattern = re.compile('^Image_(XY\\d+)_CH\\d+\\.tif$', re.IGNORECASE)
        self.condition_to_id = {'DY_ctrl': 0, 'DY_DAPT': 1, 'ST_ctrl': 2, 'ST_DAPT': 3, '2DY_ctrl': 0, '2DY_DAPT': 1, '2ST_ctrl': 2, '2ST_DAPT': 3, 'DY_GF_Pillars': 0, 'DY_GF_V_Pillars': 0, 'ST_GF_Pillars': 2}
        sample_set = set()
        possible_conditions = []
        for cond, d in self.condition_day_folders.keys():
            if d == self.input_days[0]:
                possible_conditions.append(cond)
        for cond in possible_conditions:
            day7_folder = self.condition_day_folders.get((cond, self.input_days[0]), None)
            if day7_folder is None:
                continue
            for fname in os.listdir(day7_folder):
                match = self.sample_pattern.match(fname)
                if match:
                    xy = match.group(1)
                    sample_key = f'{cond}_{xy}'
                    sample_set.add(sample_key)
        valid_samples = []
        for sk in sample_set:
            parts = sk.split('_')
            if len(parts) >= 2:
                xy = parts[-1]
                cond = '_'.join(parts[:-1])
                if self.check_valid(cond, xy):
                    valid_samples.append(sk)
            else:
                print(f'Skipping invalid sample key format: {sk}')
        self.samples = sorted(valid_samples)
        self.caption_texts = {
            condition: cell_caption(condition, self.target_day)
            if getattr(self.args, "conditioning_ablation", "none") != "text" else ""
            for condition in sorted({key.rsplit("_", 1)[0] for key in self.samples})
        }
        print(f'Found {len(self.samples)} valid samples for inference-only dataset.')

    def check_valid(self, cond, xy):
        days_needed = self.input_days
        for day in days_needed:
            folder = self.condition_day_folders.get((cond, day), None)
            if folder is None:
                return False
            for ch in self.channels:
                fname = f'Image_{xy}_{ch}.tif'
                fpath = os.path.join(folder, fname)
                if not os.path.exists(fpath):
                    return False
        return True

    def load_image(self, folder, xy, ch):
        fname = f'Image_{xy}_{ch}.tif'
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            return None
        try:
            img = read_fluorescence_image(path, ch)
            return img
        except Exception as e:
            print(f'Error loading image {path}: {e}')
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            return None
        sample_key = self.samples[idx]
        parts = sample_key.split('_')
        if len(parts) < 2:
            return None
        xy = parts[-1]
        condition = '_'.join(parts[:-1])
        pil_images_by_day = {}
        days_to_scan = self.input_days
        for day in days_to_scan:
            folder = self.condition_day_folders.get((condition, day), None)
            if folder is None:
                print(f'Missing folder for {sample_key} on {day}. Skipping sample.')
                return None
            pil_images_by_day[day] = {}
            for ch in self.channels:
                img_pil = self.load_image(folder, xy, ch)
                if img_pil is None:
                    print(f'Missing file {ch} on {day} for {sample_key}. Skipping sample.')
                    return None
                pil_images_by_day[day][ch] = img_pil
        channel_transforms = history_transforms(pil_images_by_day, self.input_days, self.channels, self.args.resolution)
        input_images_list = []
        for day in self.input_days:
            transformed_day_tensors = []
            for ch in self.channels:
                img_pil = pil_images_by_day[day][ch]
                tensor = channel_transforms[ch](img_pil)
                transformed_day_tensors.append(tensor)
            day_tensor = torch.cat(transformed_day_tensors, dim=0)
            input_images_list.append(day_tensor)
        conditioning_pixel_values = torch.stack(input_images_list, dim=0)
        text_prompt = self.caption_texts[condition]
        if getattr(self.args, 'conditioning_ablation', 'none') == 'text':
            text_prompt = ''
        input_ids = tokenize_caption(self.tokenizer, text_prompt, self.args)
        num_metadata = self.args.num_metadata
        target_metadata_values = torch.zeros(num_metadata, dtype=torch.float32)
        if num_metadata > 0:
            target_metadata_values[0] = float(self.condition_to_id.get(condition, -1))
        if num_metadata > 1:
            target_metadata_values[1] = 1.0 if 'DY' in condition else 0.0
        if num_metadata > 2:
            target_metadata_values[2] = 1.0 if 'DAPT' in condition else 0.0
        if num_metadata > 3:
            target_metadata_values[3] = float(self.target_day.replace('day', ''))
        metadata = metadata_normalize(target_metadata_values)
        conditioning_metadata_list = []
        for day_str in self.input_days:
            cond_meta_values = torch.zeros(num_metadata, dtype=torch.float32)
            if num_metadata > 0:
                cond_meta_values[0] = float(self.condition_to_id.get(condition, -1))
            if num_metadata > 1:
                cond_meta_values[1] = 1.0 if 'DY' in condition else 0.0
            if num_metadata > 2:
                cond_meta_values[2] = 1.0 if 'DAPT' in condition else 0.0
            if num_metadata > 3:
                cond_meta_values[3] = float(day_str.replace('day', ''))
            conditioning_metadata_list.append(metadata_normalize(cond_meta_values))
        conditioning_metadata = torch.stack(conditioning_metadata_list, dim=-1)
        if getattr(self.args, 'conditioning_ablation', 'none') == 'metadata':
            metadata = torch.zeros_like(metadata)
            conditioning_metadata = torch.zeros_like(conditioning_metadata)
        return {'conditioning_pixel_values': conditioning_pixel_values, 'input_ids': input_ids, 'metadata': metadata, 'conditioning_metadata': conditioning_metadata, 'sample_key': sample_key}
