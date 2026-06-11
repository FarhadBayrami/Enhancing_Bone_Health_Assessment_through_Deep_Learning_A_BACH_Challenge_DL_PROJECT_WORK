<div align="center">

# 🦴 Enhancing Bone Health Assessment through Deep Learning
### A BACH Challenge-Inspired Approach to Musculoskeletal X-ray Abnormality Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://img.shields.io/badge/AUC-84.58%25-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Accuracy-76.04%25-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/F1--Score-73.15%25-orange?style=flat-square"/>
</p>

*Deep learning model for detecting bone abnormalities in X-ray images using a modified ResNet architecture enhanced with attention mechanisms.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Comparison with State-of-the-Art](#-comparison-with-state-of-the-art)
- [Visualizations](#-visualizations)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [References](#-references)
- [Author](#-author)

---

## 🔬 Overview

This project adapts concepts from the **BACH (BreAst Cancer Histology) grand challenge** to the domain of musculoskeletal radiograph analysis. The goal is automated detection of bone abnormalities in X-ray images — a critical task in clinical workflows where timely and accurate diagnosis can significantly impact patient outcomes.

The approach leverages a custom **ResNet backbone augmented with spatial attention blocks**, trained and evaluated on the **MURA dataset** (focusing on humerus and elbow subsets).

> **Key Insight:** By combining residual connections with attention mechanisms, the model learns to focus on clinically relevant regions of radiographs, improving discrimination between normal and abnormal bone structures.

---

## 🧠 Model Architecture

The model is a modified ResNet featuring:

```
Input X-ray Image
       │
  ┌────▼────┐
  │ Conv2D  │  ← Initial feature extraction
  └────┬────┘
       │
  ┌────▼──────────────┐
  │  Residual Block   │  ← Skip connections to preserve gradient flow
  │  + Attention Gate │  ← Spatial attention to focus on anomalies
  └────┬──────────────┘
       │  (×N layers)
  ┌────▼────────────────┐
  │  Batch Norm + ReLU  │
  └────┬────────────────┘
       │
  ┌────▼──────────────────┐
  │  Global Average Pool  │  ← Reduces spatial dimensions
  └────┬──────────────────┘
       │
  ┌────▼────────┐
  │  FC + Sigmoid│  ← Binary classification output
  └─────────────┘
```

**Training Configuration:**

| Hyperparameter     | Value                          |
|--------------------|-------------------------------|
| Loss Function      | Binary Cross-Entropy           |
| Optimizer          | Adam                          |
| LR Scheduling      | ReduceLROnPlateau              |
| Data Augmentation  | Random rotation, horizontal flip |
| Hardware           | GPU recommended (Colab/Jupyter) |

---

## 📦 Dataset

**MURA** (Musculoskeletal Radiographs) — Stanford ML Group

- Binary classification: **Normal vs. Abnormal**
- Focus subsets: **Humerus** and **Elbow**
- Download: [stanfordmlgroup.github.io/competitions/mura](https://stanfordmlgroup.github.io/competitions/mura/)

> ⚠️ The MURA dataset requires registration. Update the data paths in the notebook before running.

---

## 📊 Results

Evaluated on the validation set:

| Metric        | Score    |
|---------------|----------|
| **Accuracy**  | 76.04%   |
| **Precision** | 80.34%   |
| **Recall**    | 67.14%   |
| **F1-Score**  | 73.15%   |
| **AUC**       | 84.58%   |

---

## 🏆 Comparison with State-of-the-Art

| Model                  | Accuracy | Notes                          |
|------------------------|----------|--------------------------------|
| **This Model**         | 76.04%   | ResNet + Attention, MURA subset |
| Marami et al. (2018)   | 84%      | Full MURA dataset               |
| Kohl et al. (2018)     | 83%      | Full MURA dataset               |
| Wang et al. (2018a)    | 83%      | Full MURA dataset               |

> **Note:** Comparisons are indicative. Direct equivalence depends on dataset splits, body part subsets, and evaluation protocols.

---

## 📈 Visualizations

All plots are included in the [project report PDF](./Enhancing_Bone_Health_Assessment_through_Deep_Learning_A_BACH_Challenge.pdf).

| Plot | Description |
|------|-------------|
| 📉 **Loss Curve** | Initial spikes, stabilises at ~0.5–0.75 after ~40 epochs |
| 📈 **Accuracy Curve** | Converges to ~70–75% on validation |
| 🔲 **Confusion Matrix** | Reasonable class separation between normal/abnormal |
| 📐 **ROC Curve** | AUC = 0.85, good discrimination ability |
| 🎯 **Precision-Recall** | High initial precision with gradual degradation |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy opencv-python scikit-learn matplotlib keras
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/FarhadBayrami/Enhancing_Bone_Health_Assessment_through_Deep_Learning.git
cd Enhancing_Bone_Health_Assessment_through_Deep_Learning

# 2. Install dependencies
pip install torch torchvision numpy opencv-python scikit-learn matplotlib keras

# 3. Download the MURA dataset
#    → https://stanfordmlgroup.github.io/competitions/mura/
#    → Update file paths in the notebook accordingly

# 4. Launch the notebook
jupyter notebook "ResAttention_BACH_Challenege (1).ipynb"
```

> 💡 **Tip:** Run on Google Colab with a GPU runtime for faster training. Change the runtime type under *Runtime → Change runtime type → GPU*.

---

## 📁 Project Structure

```
📦 Enhancing_Bone_Health_Assessment_through_Deep_Learning
 ┣ 📓 ResAttention_BACH_Challenege (1).ipynb   ← Full pipeline: data, model, training, eval
 ┣ 📄 Enhancing_Bone_Health_Assessment_...pdf  ← Complete project report
 ┗ 📝 README.md
```

---

## 🔮 Future Work

- [ ] Fine-tune architecture depth and width for MURA full dataset
- [ ] Incorporate transformer-based attention (e.g., Vision Transformer)
- [ ] Expand to additional body parts (wrist, knee, shoulder)
- [ ] Grad-CAM visualisations for clinical explainability
- [ ] Clinical validation with radiologist review

---

## 📚 References

1. Aresta, G. et al. — *BACH: Grand challenge on breast cancer histology images*, Medical Image Analysis, 2019.
2. Rajpurkar, P. et al. — *MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs*, 2018.
3. Marami, B. et al. (2018), Kohl, S. et al. (2018), Wang, X. et al. (2018) — Comparative baselines on MURA.

---

## 👤 Author

**Farhad Bayrami**
MSc Student — University of Bologna
📧 [farhad.bayrami@studio.unibo.it](mailto:farhad.bayrami@studio.unibo.it)
🔗 [GitHub](https://github.com/FarhadBayrami)

---

<div align="center">
  <sub>Built with ❤️ as part of a Deep Learning course project at the University of Bologna</sub>
</div>