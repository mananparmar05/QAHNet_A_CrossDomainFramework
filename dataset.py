
"""
PAD-UFES-20 Dataset for QAHNet.

Handles:
  - Loading images from imgs_part_1/2/3
  - Parsing all 26 metadata columns
  - Runtime augmentation + degradation simulation
  - Quality score generation for FiLM QTok training
  - Stratified train/val/test splitting
"""
import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from models.metadata_encoder import (
    encode_metadata_row, INFECTION_MAP, SEVERITY_MAP, DIAGNOSTIC_MAP
)


class DegradationSimulator:
    """
    Simulates real-world image degradation (blur, noise, JPEG compression).
    Returns the degraded image and a quality score in [0, 1].
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            (degraded_img, quality_score)
        """
        if random.random() > self.prob:
            return img, 1.0  # clean image

        quality = 1.0
        degradations = random.sample(
            ['blur', 'noise', 'jpeg', 'color'],
            k=random.randint(1, 2)
        )

        for deg in degradations:
            if deg == 'blur':
                radius = random.uniform(0.5, 3.0)
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))
                quality *= max(0.3, 1.0 - radius / 4.0)

            elif deg == 'noise':
                arr = np.array(img).astype(np.float32)
                sigma = random.uniform(5, 30)
                noise = np.random.normal(0, sigma, arr.shape)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
                quality *= max(0.3, 1.0 - sigma / 40.0)

            elif deg == 'jpeg':
                import io
                q = random.randint(10, 50)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=q)
                buffer.seek(0)
                img = Image.open(buffer).convert('RGB')
                quality *= max(0.3, q / 100.0)

            elif deg == 'color':
                arr = np.array(img).astype(np.float32)
                factor = random.uniform(0.7, 1.3)
                arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
                quality *= 0.9

        return img, max(0.0, min(1.0, quality))


class PADDataset(Dataset):
    """
    PAD-UFES-20 Dataset for QAHNet training.

    Args:
        data_dir: path to PADUFES folder (with imgs_part_1/2/3 and metadata.csv)
        split: 'train', 'val', or 'test'
        transform: torchvision transforms
        degrade_prob: probability of applying degradation (train only)
        seed: random seed for reproducible splits
    """

    def __init__(self, data_dir, split='train', transform=None,
                 degrade_prob=0.5, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.degrader = DegradationSimulator(
            prob=degrade_prob if split == 'train' else 0.0
        )

        # Load metadata
        csv_path = os.path.join(data_dir, 'metadata.csv')
        self.df = pd.read_csv(csv_path)

        # Build image path lookup
        self.img_dir_map = {}
        for part in ['imgs_part_1', 'imgs_part_2', 'imgs_part_3']:
            part_dir = os.path.join(data_dir, part)
            if os.path.isdir(part_dir):
                # Handle nested folders (imgs_part_X/imgs_part_X/)
                inner = os.path.join(part_dir, part)
                if os.path.isdir(inner):
                    part_dir = inner
                for fname in os.listdir(part_dir):
                    if fname.lower().endswith('.png'):
                        self.img_dir_map[fname] = os.path.join(part_dir, fname)

        # Filter to rows with valid images and known diagnostics
        valid_diags = set(DIAGNOSTIC_MAP.keys())
        self.df = self.df[
            self.df['img_id'].isin(self.img_dir_map) &
            self.df['diagnostic'].isin(valid_diags)
        ].reset_index(drop=True)

        # Create stratified split based on diagnostic label
        self._create_split(seed)

        print(f"[{split.upper()}] {len(self.indices)} samples")

    def _create_split(self, seed):
        """70/15/15 stratified train/val/test split."""
        labels = self.df['diagnostic'].values
        indices = np.arange(len(self.df))

        # First split: 70% train, 30% temp
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.30, random_state=seed,
            stratify=labels
        )

        # Second split: 50/50 of temp = 15% val, 15% test
        temp_labels = labels[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=seed,
            stratify=temp_labels
        )

        split_map = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        self.indices = split_map[self.split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]

        # Load image
        img_id = row['img_id']
        img_path = self.img_dir_map[img_id]
        img = Image.open(img_path).convert('RGB')

        # Apply degradation (returns image + quality score)
        img, quality_score = self.degrader(img)

        # Apply transforms (resize, normalize, augmentation)
        if self.transform:
            img = self.transform(img)

        # Labels
        diag = row['diagnostic']
        infection = INFECTION_MAP[diag]
        severity = SEVERITY_MAP[diag]

        # Metadata
        metadata = encode_metadata_row(row)

        return {
            'image': img,
            'infection': torch.tensor(infection, dtype=torch.float32),
            'severity': torch.tensor(severity, dtype=torch.long),
            'quality_score': torch.tensor(quality_score, dtype=torch.float32),
            'metadata': metadata,
            'diagnostic': diag,
            'img_id': img_id,
        }


def collate_fn(batch):
    """Custom collate to handle metadata dict."""
    images = torch.stack([b['image'] for b in batch])
    infection = torch.stack([b['infection'] for b in batch])
    severity = torch.stack([b['severity'] for b in batch])
    quality = torch.stack([b['quality_score'] for b in batch])

    # Collate metadata dict
    meta_keys = batch[0]['metadata'].keys()
    metadata = {}
    for key in meta_keys:
        metadata[key] = torch.stack([b['metadata'][key] for b in batch])

    return {
        'image': images,
        'infection': infection,
        'severity': severity,
        'quality_score': quality,
        'metadata': metadata,
    }


def get_transforms(split='train', img_size=160):
    """Get transforms for train/val/test. Default 160px to save memory."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


def create_dataloaders(data_dir, batch_size=32, num_workers=4, seed=42):
    """Create train/val/test dataloaders."""
    loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = PADDataset(
            data_dir=data_dir,
            split=split,
            transform=get_transforms(split),
            degrade_prob=0.5 if split == 'train' else 0.0,
            seed=seed,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=(split == 'train'),
        )
    return loaders