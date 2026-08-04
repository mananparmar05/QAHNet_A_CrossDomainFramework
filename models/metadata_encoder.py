"""
Clinical Metadata Encoder Module.

Novel contribution C3: Encodes patient clinical metadata (age, sex,
Fitzpatrick, region, diameter, symptoms) into learnable tokens that
participate in transformer self-attention alongside visual tokens.
"""
import torch
import torch.nn as nn


# Categorical mappings for PAD-UFES-20
GENDER_MAP = {"MALE": 0, "FEMALE": 1}

REGION_MAP = {
    "FACE": 0, "NOSE": 1, "NECK": 2, "CHEST": 3, "BACK": 4,
    "ARM": 5, "FOREARM": 6, "HAND": 7, "THIGH": 8, "FOOT": 9,
    "EAR": 10, "SCALP": 11, "ABDOMEN": 12, "LIP": 13, "LEG": 14,
}

DIAGNOSTIC_MAP = {
    "ACK": 0, "BCC": 1, "MEL": 2, "NEV": 3, "SCC": 4, "SEK": 5,
}

# Infection: BCC, MEL, SCC = 1 (malignant/precancerous); ACK, NEV, SEK = 0
INFECTION_MAP = {"ACK": 0, "BCC": 1, "MEL": 1, "NEV": 0, "SCC": 1, "SEK": 0}

# Severity: 0=mild (NEV, SEK), 1=moderate (ACK, BCC), 2=severe (SCC, MEL)
SEVERITY_MAP = {"NEV": 0, "SEK": 0, "ACK": 1, "BCC": 1, "SCC": 2, "MEL": 2}


class MetadataEncoder(nn.Module):
    """
    Encodes clinical metadata into K learnable tokens.

    Features encoded:
        - age (numeric, normalized)
        - gender (categorical: male/female)
        - fitspatrick (ordinal: 1-6)
        - region (categorical: 15 body locations)
        - diameter (numeric: max of diameter_1, diameter_2)
        - itch, grew, hurt, changed, bleed (binary symptoms)
        - elevation (binary)

    Total: 11 features -> 11 tokens of embed_dim each
    """

    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # Numeric feature encoders (age, diameter)
        self.age_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

        self.diameter_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

        # Categorical embeddings
        self.gender_embed = nn.Embedding(3, embed_dim)       # male, female, unknown
        self.fitspatrick_embed = nn.Embedding(8, embed_dim)  # 0-6 + unknown(7)
        self.region_embed = nn.Embedding(16, embed_dim)      # 15 regions + unknown

        # Binary symptom embeddings (each: 0=no, 1=yes, 2=unknown)
        self.itch_embed = nn.Embedding(3, embed_dim)
        self.grew_embed = nn.Embedding(3, embed_dim)
        self.hurt_embed = nn.Embedding(3, embed_dim)
        self.changed_embed = nn.Embedding(3, embed_dim)
        self.bleed_embed = nn.Embedding(3, embed_dim)
        self.elevation_embed = nn.Embedding(3, embed_dim)

        # Positional embeddings for metadata tokens (11 tokens)
        self.meta_pos = nn.Parameter(torch.randn(1, 11, embed_dim) * 0.02)

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, metadata):
        """
        Args:
            metadata: dict with keys:
                'age': [B, 1] float (normalized 0-1)
                'gender': [B] long
                'fitspatrick': [B] long
                'region': [B] long
                'diameter': [B, 1] float (normalized 0-1)
                'itch': [B] long
                'grew': [B] long
                'hurt': [B] long
                'changed': [B] long
                'bleed': [B] long
                'elevation': [B] long

        Returns:
            meta_tokens: [B, 11, embed_dim]
        """
        tokens = []

        # Numeric features
        tokens.append(self.age_encoder(metadata['age']))           # [B, D]
        tokens.append(self.diameter_encoder(metadata['diameter']))  # [B, D]

        # Categorical features
        tokens.append(self.gender_embed(metadata['gender']))          # [B, D]
        tokens.append(self.fitspatrick_embed(metadata['fitspatrick']))  # [B, D]
        tokens.append(self.region_embed(metadata['region']))          # [B, D]

        # Binary symptoms
        tokens.append(self.itch_embed(metadata['itch']))       # [B, D]
        tokens.append(self.grew_embed(metadata['grew']))       # [B, D]
        tokens.append(self.hurt_embed(metadata['hurt']))       # [B, D]
        tokens.append(self.changed_embed(metadata['changed']))  # [B, D]
        tokens.append(self.bleed_embed(metadata['bleed']))     # [B, D]
        tokens.append(self.elevation_embed(metadata['elevation']))  # [B, D]

        # Stack into sequence: [B, 11, D]
        meta_tokens = torch.stack(tokens, dim=1)

        # Add positional embeddings + normalize
        meta_tokens = self.norm(meta_tokens + self.meta_pos)

        return meta_tokens


def encode_metadata_row(row):
    """
    Convert a single metadata CSV row into model-ready tensors.

    Args:
        row: pandas Series from metadata.csv

    Returns:
        dict of tensors (before batching)
    """
    def safe_bool(val):
        """Convert TRUE/FALSE/UNK/empty to 0/1/2."""
        if isinstance(val, bool):
            return 1 if val else 0
        val = str(val).strip().upper()
        if val in ('TRUE', '1', 'YES'):
            return 1
        elif val in ('FALSE', '0', 'NO'):
            return 0
        return 2  # unknown / UNK / empty

    def safe_float(val, default=0.0):
        try:
            v = float(val)
            return v if not (v != v) else default  # handle NaN
        except (ValueError, TypeError):
            return default

    # Age: normalize to [0, 1] (max age ~100)
    age = safe_float(row.get('age', 0), 0.0) / 100.0

    # Gender
    gender_str = str(row.get('gender', '')).strip().upper()
    gender = GENDER_MAP.get(gender_str, 2)  # 2 = unknown

    # Fitzpatrick skin type (1-6, 0=unknown)
    fitz = safe_float(row.get('fitspatrick', 0), 0.0)
    fitz = int(min(max(fitz, 0), 6))
    if fitz == 0:
        fitz = 7  # unknown

    # Region
    region_str = str(row.get('region', '')).strip().upper()
    region = REGION_MAP.get(region_str, 15)  # 15 = unknown

    # Diameter: max of diameter_1 and diameter_2, normalize (max ~60mm)
    d1 = safe_float(row.get('diameter_1', 0), 0.0)
    d2 = safe_float(row.get('diameter_2', 0), 0.0)
    diameter = max(d1, d2) / 60.0

    # Binary symptoms
    itch = safe_bool(row.get('itch', ''))
    grew = safe_bool(row.get('grew', ''))
    hurt = safe_bool(row.get('hurt', ''))
    changed = safe_bool(row.get('changed', ''))
    bleed = safe_bool(row.get('bleed', ''))
    elevation = safe_bool(row.get('elevation', ''))

    return {
        'age': torch.tensor([age], dtype=torch.float32),
        'gender': torch.tensor(gender, dtype=torch.long),
        'fitspatrick': torch.tensor(fitz, dtype=torch.long),
        'region': torch.tensor(region, dtype=torch.long),
        'diameter': torch.tensor([diameter], dtype=torch.float32),
        'itch': torch.tensor(itch, dtype=torch.long),
        'grew': torch.tensor(grew, dtype=torch.long),
        'hurt': torch.tensor(hurt, dtype=torch.long),
        'changed': torch.tensor(changed, dtype=torch.long),
        'bleed': torch.tensor(bleed, dtype=torch.long),
        'elevation': torch.tensor(elevation, dtype=torch.long),
    }
