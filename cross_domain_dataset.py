"""
Cross-Domain Dataset Adapter for QAHNet.

Adapts ISIC 2019 dataset to match PAD-UFES-20 format exactly,
so QAHNet (trained on PAD) can be evaluated on ISIC with zero retraining.

Key challenge: ISIC has only ~5 metadata fields vs PAD's 26.
Strategy: Missing fields → index 2 (unknown) for categoricals,
          -1.0 normalized for numerics (out-of-distribution signal).

Usage:
    dataset = ISICDataset(data_dir='./ISIC2019')
    loader  = DataLoader(dataset, batch_size=16, collate_fn=isic_collate_fn)

Directory structure expected:
    ISIC2019/
        ISIC_2019_Training_Input/      ← .jpg images
        ISIC_2019_Training_GroundTruth.csv
        ISIC_2019_Training_Metadata.csv
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ── Label mappings ─────────────────────────────────────────────────────────────

# ISIC 8 classes → infection (benign=0, malignant=1)
ISIC_INFECTION_MAP = {
    'MEL':  1,   # Melanoma — malignant
    'BCC':  1,   # Basal Cell Carcinoma — malignant
    'SCC':  1,   # Squamous Cell Carcinoma — malignant
    'AK':   1,   # Actinic Keratosis — pre-malignant
    'NV':   0,   # Melanocytic Nevus — benign
    'BKL':  0,   # Benign Keratosis — benign
    'DF':   0,   # Dermatofibroma — benign
    'VASC': 0,   # Vascular Lesion — benign
}

# ISIC classes → severity (0=mild, 1=moderate, 2=severe)
ISIC_SEVERITY_MAP = {
    'MEL':  2,   # severe
    'SCC':  2,   # severe
    'BCC':  1,   # moderate
    'AK':   1,   # moderate (pre-cancerous)
    'NV':   0,   # mild
    'BKL':  0,   # mild
    'DF':   0,   # mild
    'VASC': 0,   # mild
}

# ISIC anatomical site → PAD-UFES-20 region index
# PAD regions: FACE=0, NOSE=1, NECK=2, CHEST=3, BACK=4,
#              ARM=5, FOREARM=6, HAND=7, THIGH=8, FOOT=9,
#              EAR=10, SCALP=11, ABDOMEN=12, LIP=13, LEG=14, unknown=15
ISIC_REGION_MAP = {
    'head/neck':       2,   # → NECK (closest general match)
    'upper extremity': 5,   # → ARM
    'lower extremity': 14,  # → LEG
    'torso':           3,   # → CHEST (front torso)
    'palms/soles':     7,   # → HAND
    'oral/genital':    13,  # → LIP (closest available)
    'anterior torso':  3,   # → CHEST
    'posterior torso': 4,   # → BACK
    'lateral torso':   3,   # → CHEST
}


def build_isic_metadata_tensor(row):
    """
    Map one ISIC metadata row to the exact same tensor format
    as encode_metadata_row() in metadata_encoder.py.

    Missing ISIC fields use:
        - Categoricals: index 2 (unknown slot in each embedding)
        - Numerics: 0.5 (neutral, mid-range — better than -1 for MLP inputs)

    Returns:
        dict of tensors matching PAD-UFES-20 metadata format
    """
    # ── Age ──
    age_raw = row.get('age_approx', None)
    try:
        age = float(age_raw) / 100.0
        age = max(0.0, min(1.0, age))
    except (TypeError, ValueError):
        age = 0.5  # unknown → mid-range

    # ── Gender → matches GENDER_MAP: MALE=0, FEMALE=1, unknown=2 ──
    sex = str(row.get('sex', '')).strip().lower()
    if sex == 'male':
        gender = 0
    elif sex == 'female':
        gender = 1
    else:
        gender = 2  # unknown

    # ── Fitzpatrick: ISIC doesn't have it → unknown=7 (PAD encoder uses 7) ──
    fitspatrick = 7  # unknown

    # ── Region ──
    site = str(row.get('anatom_site_general_challenge', '')).strip().lower()
    region = ISIC_REGION_MAP.get(site, 15)  # 15 = unknown

    # ── Diameter: ISIC doesn't have it → 0.5 (neutral) ──
    diameter = 0.5

    # ── All symptoms unknown (ISIC has none of these) ──
    # unknown index = 2 in PAD encoder embeddings
    itch      = 2
    grew      = 2
    hurt      = 2
    changed   = 2
    bleed     = 2
    elevation = 2

    return {
        'age':        torch.tensor([age],        dtype=torch.float32),
        'gender':     torch.tensor(gender,       dtype=torch.long),
        'fitspatrick':torch.tensor(fitspatrick,  dtype=torch.long),
        'region':     torch.tensor(region,       dtype=torch.long),
        'diameter':   torch.tensor([diameter],   dtype=torch.float32),
        'itch':       torch.tensor(itch,         dtype=torch.long),
        'grew':       torch.tensor(grew,         dtype=torch.long),
        'hurt':       torch.tensor(hurt,         dtype=torch.long),
        'changed':    torch.tensor(changed,      dtype=torch.long),
        'bleed':      torch.tensor(bleed,        dtype=torch.long),
        'elevation':  torch.tensor(elevation,    dtype=torch.long),
    }


class ISICDataset(Dataset):
    """
    ISIC 2019 dataset adapted for QAHNet cross-domain evaluation.

    Args:
        data_dir: path to ISIC2019 folder
        transform: torchvision transforms (use get_isic_transform())
        max_samples: optional cap for quick testing (None = all)
    """

    def __init__(self, data_dir, transform=None, max_samples=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.img_dir = os.path.join(data_dir, 'ISIC_2019_Training_Input')

        # ── Load ground truth ──
        gt_path = os.path.join(data_dir, 'ISIC_2019_Training_GroundTruth.csv')
        gt_df = pd.read_csv(gt_path)

        # One-hot → class label
        label_cols = [c for c in ['MEL','NV','BCC','AK','BKL','DF','VASC','SCC']
                      if c in gt_df.columns]
        gt_df['label'] = gt_df[label_cols].idxmax(axis=1)

        # ── Load metadata ──
        meta_path = os.path.join(data_dir, 'ISIC_2019_Training_Metadata.csv')
        if os.path.exists(meta_path):
            meta_df = pd.read_csv(meta_path)
            self.df = gt_df.merge(meta_df, on='image', how='left')
        else:
            print("[WARNING] ISIC metadata CSV not found — all metadata will be unknown")
            self.df = gt_df

        # ── Filter to known label classes ──
        self.df = self.df[
            self.df['label'].isin(ISIC_INFECTION_MAP)
        ].reset_index(drop=True)

        # ── Optional cap ──
        if max_samples is not None:
            self.df = self.df.sample(
                n=min(max_samples, len(self.df)), random_state=42
            ).reset_index(drop=True)

        print(f"[ISIC Cross-Domain] {len(self.df)} samples loaded")
        print(f"  Label distribution:\n{self.df['label'].value_counts().to_string()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Image ──
        img_name = row['image']
        # Try .jpg then .JPG then no extension
        for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
            img_path = os.path.join(self.img_dir, img_name + ext)
            if os.path.exists(img_path):
                break
        else:
            img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # ── Labels ──
        label = row['label']
        infection = ISIC_INFECTION_MAP[label]
        severity  = ISIC_SEVERITY_MAP[label]

        # ── Metadata ──
        metadata = build_isic_metadata_tensor(row)

        return {
            'image':         img,
            'infection':     torch.tensor(infection,  dtype=torch.float32),
            'severity':      torch.tensor(severity,   dtype=torch.long),
            'quality_score': torch.tensor(1.0,        dtype=torch.float32),
            'metadata':      metadata,
            'label':         label,
            'img_id':        img_name,
        }


def isic_collate_fn(batch):
    """Collate function — matches PAD-UFES-20 collate_fn signature."""
    images    = torch.stack([b['image']         for b in batch])
    infection = torch.stack([b['infection']     for b in batch])
    severity  = torch.stack([b['severity']      for b in batch])
    quality   = torch.stack([b['quality_score'] for b in batch])

    meta_keys = batch[0]['metadata'].keys()
    metadata  = {k: torch.stack([b['metadata'][k] for b in batch]) for k in meta_keys}

    return {
        'image':         images,
        'infection':     infection,
        'severity':      severity,
        'quality_score': quality,
        'metadata':      metadata,
    }


def get_isic_transform(img_size=160):
    """
    Same eval transform as PAD-UFES-20 test split.
    ISIC images are dermatoscope (higher res) — we resize to match training.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_isic_loader(data_dir, batch_size=16, num_workers=0,
                       img_size=160, max_samples=None):
    """
    Create a ready-to-use ISIC DataLoader for cross-domain evaluation.

    Args:
        data_dir:    path to ISIC2019 folder
        batch_size:  keep low (16) for 8GB RAM
        num_workers: 0 on Mac to avoid multiprocessing issues
        img_size:    must match training (160)
        max_samples: cap for quick testing

    Returns:
        DataLoader
    """
    dataset = ISICDataset(
        data_dir=data_dir,
        transform=get_isic_transform(img_size),
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=isic_collate_fn,
        pin_memory=False,
    )
