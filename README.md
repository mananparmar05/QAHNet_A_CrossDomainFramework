# QAHNet: Quality-Aware Hybrid Network for Multi-Task Disease Triage

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Domain: Medical AI & Computer Vision](https://img.shields.io/badge/Domain-Medical%20AI%20%26%20Vision-green.svg)]()

Official PyTorch implementation of **QAHNet** (*Quality-Aware Hybrid Network*), a novel cross-domain multi-task deep learning architecture designed for robust disease classification, severity grading, and image quality assessment under severe real-world image degradations (Gaussian blur, additive noise, JPEG compression).

---

## 🌟 Key Features & Architectural Contributions

QAHNet addresses the critical vulnerability of standard deep learning models when operating on uncontrolled, mobile-acquired diagnostic images.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 Input Diagnostic Image                 │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                  ┌───────────┴───────────┐
                                  │ EfficientNet-B0       |
                                  │ Backbone              |
                                  └───────────┬───────────┘
                                              │ 7x7x1280 Feature Map
                                  ┌───────────┴───────────┐
                                  │   Patch Tokenizer     │
                                  └───────────┬───────────┘
                                              │ 49 Patch Tokens
                                              │
  ┌──────────────────────┐        ┌───────────┴───────────┐        ┌──────────────────────┐
  │ Estimated Quality q  ├───────►│ FiLM QTok Generator   ├───────►│ Scale (γ) & Shift (β)│
  └──────────────────────┘        └───────────┬───────────┘        └───────────┬──────────┘
                                              │ Quality Token (t_q)            │ Modulates
  ┌──────────────────────┐        ┌───────────┴───────────┐                    │ Attention
  │  Clinical Metadata   ├───────►│   Metadata Encoder    │                    │ Weights
  └──────────────────────┘        └───────────┬───────────┘                    │
                                              │ 11 Meta Tokens                 │
                                              ▼                                ▼
                        ┌─────────────────────────────────────────────────────────────┐
                        │   Quality-Conditioned Transformer Encoder (L=2, H=4)        │
                        └─────────────────────────────┬───────────────────────────────┘
                                                      │ CLS Token Output
                                    ┌─────────────────┼─────────────────┐
                                    ▼                 ▼                 ▼
                             ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
                             │Classification│  │  Severity   │   │   Quality   │
                             │    Head     │   │    Head     │   │    Head     │
                             └─────────────┘   └─────────────┘   └─────────────┘
```

1. **Feature-wise Linear Modulation (FiLM) Quality Token Generator (`C1`)**:
   Generates per-attention-head scaling ($\gamma$) and shifting ($\beta$) vectors based on input image quality to dynamically adapt transformer attention patterns under image degradation:
   $$\gamma = \sigma(W_\gamma e_q) \in [0, 1]^H, \quad \beta = \tanh(W_\beta e_q) \in [-1, 1]^H$$
   $$A_{\text{modulated}} = \gamma \odot \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) + \beta$$

2. **Early Multimodal Clinical Metadata Tokenization (`C3`)**:
   Projects 11 clinical metadata attributes (patient age, sex, Fitzpatrick skin type, lesion region, max diameter, and binary symptoms like itch, growth, bleeding, pain, elevation) into learnable embedding tokens that interact directly with visual tokens via full self-attention.

3. **Joint Multi-Task Objective (`C2`)**:
   Simultaneously optimizes disease classification (binary/multi-class infection), 3-tier severity grading (Mild, Moderate, Severe), and quality score regression using a unified multi-task loss:
   $$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{CE}}(\hat{y}_{\text{cls}}, y_{\text{cls}}) + \lambda_2 \mathcal{L}_{\text{CE}}(\hat{y}_{\text{sev}}, y_{\text{sev}}) + \lambda_3 \mathcal{L}_{\text{MSE}}(\hat{q}, q)$$
   with weighting factors $\lambda_1 = 1.0$, $\lambda_2 = 0.5$, $\lambda_3 = 0.3$.

4. **Cross-Domain Generalization**:
   Evaluated across human dermatological datasets (**PAD-UFES-20**, **ISIC 2019**) and agricultural plant leaf disease benchmarks (**PlantDoc / PlantVillage**).

---

## 📊 Experimental Results & Ablation Study

Component contribution analysis evaluated across 6 model variants on held-out test splits under synthetic mobile image degradation:

| Rank | Model Variant | Architecture Configuration | Infection F1 | Infection AUC | Severity F1 | Parameters |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 🥇 **1st** | **Variant E** | **FiLM + Clinical Metadata** | **0.884** | **0.960** | **0.859** | ~18.5M |
| 🥈 **2nd** | **Variant F** | **Full QAHNet (FiLM + Meta + Quality Head)** | **0.871** | **0.948** | **0.836** | ~18.6M |
| 3rd | Variant C | Passive Quality Token (No FiLM) | 0.761 | 0.846 | 0.784 | ~18.3M |
| 4th | Variant A | CNN-only Baseline (EfficientNet-B0) | 0.753 | 0.858 | 0.809 | ~13.3M |
| 5th | Variant B | CNN + Plain Transformer Encoder | 0.748 | 0.854 | 0.784 | ~18.1M |
| 6th | Variant D | FiLM Quality Token Only (No Metadata) | 0.725 | 0.837 | 0.760 | ~18.3M |

### 🛡️ Robustness Under Image Degradation
Evaluating performance under synthetic mobile imaging corruptions:

| Degradation Condition | Blur ($\sigma=2.0$) | Noise ($\sigma=25$) | JPEG ($Q=10$) | Clean Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **QAHNet (FiLM + Metadata)** | **0.862** | **0.841** | **0.855** | **0.884** |
| CNN-only Baseline | 0.710 | 0.684 | 0.715 | 0.753 |
| Plain CNN-Transformer | 0.695 | 0.670 | 0.702 | 0.748 |

---

## 📂 Repository Structure

```
QAHnet/
├── checkpoints/             # Trained variant outputs, curves, and metrics
│   ├── variant_A/           # CNN-only baseline results & history
│   ├── variant_B/           # CNN + Transformer results
│   ├── variant_C/           # Passive QTok results
│   ├── variant_D/           # FiLM-only results
│   ├── variant_E/           # FiLM + Metadata (Best overall performer)
│   └── variant_F/           # Full QAHNet results
├── models/                  # Core PyTorch model architectures
│   ├── __init__.py          # Package exports
│   ├── film_module.py       # FiLM Quality Token generator (C1)
│   ├── metadata_encoder.py  # 11-attribute clinical metadata encoder (C3)
│   └── qahnet.py            # Main QAHNet & Ablation model factory
├── trial imges/             # Sample clinical trial images
├── cross_domain_dataset.py  # ISIC 2019 cross-domain data adapter
├── dataset.py               # PAD-UFES-20 dataset loader & degradation simulator
├── evaluate.py              # Single & multi-variant evaluation suite
├── evaluate_cross_domain.py # Zero-shot cross-domain evaluation (PAD -> ISIC)
├── run_ablation.py          # Automated ablation experiment runner
├── train.py                 # Multi-task training pipeline with differential LRs
├── requirements.txt         # Project dependencies
├── HybridmodelRP.pdf        # Research paper documentation
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/QAHnet.git
cd QAHnet
pip install -r requirements.txt
```

### 2. Dataset Preparation
Download the **PAD-UFES-20** dataset or **ISIC 2019** dataset and extract to your workspace directory:
```
PADUFES/
├── imgs_part_1/
├── imgs_part_2/
├── imgs_part_3/
└── metadata.csv
```

### 3. Training
Train the full QAHNet model (Variant F) or any ablation variant (`A` through `F`):

```bash
# Train Full QAHNet (Variant F)
python train.py --variant F --data_dir ./PADUFES --batch_size 32 --epochs 100

# Train Ablation Variant E (FiLM + Metadata)
python train.py --variant E --data_dir ./PADUFES --batch_size 32 --epochs 100

# Train CNN-only Baseline (Variant A)
python train.py --variant A --data_dir ./PADUFES --batch_size 32 --epochs 100
```

Key training arguments:
- `--variant`: Choice of model variant (`A`, `B`, `C`, `D`, `E`, `F`).
- `--lr_cnn`: Learning rate for CNN backbone (default: `1e-5`).
- `--lr_transformer`: Learning rate for transformer & FiLM module (default: `1e-4`).
- `--lr_heads`: Learning rate for classification & prediction heads (default: `3e-4`).
- `--freeze_epochs`: Initial epochs with frozen CNN backbone for stable warmup (default: `5`).

### 4. Running Full Ablation Suite
To train and benchmark all 6 model variants sequentially:
```bash
python run_ablation.py --data_dir ./PADUFES --output_dir ./checkpoints
```

### 5. Evaluation & Robustness Benchmarking
Generate the full ablation comparison table and test degradation robustness:
```bash
# Generate evaluation summary across all checkpoints
python evaluate.py --data_dir ./PADUFES --output_dir ./checkpoints

# Evaluate specific variant under synthetic image degradations
python evaluate.py --data_dir ./PADUFES --output_dir ./checkpoints --variant E
```

### 6. Zero-Shot Cross-Domain Evaluation (PAD $\rightarrow$ ISIC 2019)
Evaluate trained QAHNet checkpoints on the ISIC 2019 dataset without fine-tuning:
```bash
python evaluate_cross_domain.py \
    --isic_dir ./ISIC2019 \
    --pad_dir ./PADUFES \
    --variant E \
    --output_dir ./cross_domain_results
```

---

## Collaborators
Nachiket Jain ([GitHub](https://github.com/Nachiket-Jain)) ([Mail](mailto:nachiketjain5@gmail.com))
<br>
Manan Parmar ([GitHub](https://github.com/mananparmar05)) ([Mail](mailto:mananparmar05@gmail.com))
<br>
Madhav Patel ([GitHub](github.com/Madhavpatel24)) ([Mail](mailto:2425madhavp@gmail.com))
<br>

## 🛠️ Citation & Acknowledgments

If you find this codebase or research implementation useful in your work, please cite:

```bibtex
@article{machado2025qahnet,
  title={QAHNet: Quality-Aware Hybrid Network for Cross-Domain Disease Classification and Severity Assessment},
  author={Machado, Sweedle and others},
  journal={Procedia Computer Science},
  year={2025}
}
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).
