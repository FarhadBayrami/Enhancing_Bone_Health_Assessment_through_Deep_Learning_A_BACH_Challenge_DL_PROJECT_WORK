# Enhancing_Bone_Health_Assessment_through_Deep_Learning_A_BACH_Challenge_DL_PROJECT_WORK
Deep learning model for detecting bone abnormalities in X-ray images using a modified ResNet with attention mechanisms.

## Overview
This project adapts concepts from medical image analysis challenges (inspired by BACH histology challenge) to musculoskeletal X-ray abnormality detection.  
- **Dataset**: MURA (focus on subsets like humerus/elbow).  
- **Model**: ResNet with residual + attention blocks, batch norm, ReLU, global average pooling.  
- **Training**: Binary cross-entropy, Adam optimizer, LR scheduling, data augmentation (rotation, flip).  
- **Results** (validation set):  
  - Accuracy: **76.04%**  
  - Precision: **80.34%**  
  - Recall: **67.14%**  
  - F1-Score: **73.15%**  
  - AUC: **84.58%**

## Comparison with Related Works
| Model                  | Accuracy |
|------------------------|----------|
| This model             | 76.04%  |
| Marami et al. (2018)   | 84%     |
| Kohl et al. (2018)     | 83%     |
| Wang et al. (2018a)    | 83%     |

(Note: Comparisons are indicative; direct equivalence depends on exact dataset splits and body parts.)

## Files
- `ResAttention_BACH_Challenege (1).ipynb`: Full code (data loading, model, training, evaluation, plots).  
- `Enhancing_Bone_Health_Assessment_through_Deep_Learning_A_BACH_Challenge.pdf`: Complete project report with methodology, results, plots, and discussion.

## Visualizations (from report)
- Training/Validation Loss: Initial spikes, stabilizes ~0.5–0.75 after ~40 epochs.  
- Training/Validation Accuracy: Converges to ~70–75% range.  
- Confusion Matrix: Shows reasonable class separation.  
- ROC Curve: AUC = 0.85 (good discrimination).  
- Precision-Recall Curve: High initial precision, gradual drop.

## How to Run
1. Clone repo: `git clone https://github.com/FarhadBayrami/Enhancing_Bone_Health_Assessment_through_Deep_Learning_A_BACH_Challenge_DL_PROJECT_WORK.git`  
2. Install deps: `pip install torch torchvision numpy opencv-python scikit-learn matplotlib keras`  
3. Download MURA dataset (from Stanford ML Group).  
4. Update paths in notebook → run cells sequentially (best in Colab/Jupyter with GPU).

## Future Work
Fine-tune architecture, add advanced attention, incorporate more data, clinical validation.

## Reference
- BACH: Grand challenge on breast cancer histology images, *Medical Image Analysis*, 2019.  
- MURA dataset: Rajpurkar et al., "MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs", 2017/2018.

Author: Farhad Bayrami – farhad.bayrami@studio.unibo.it
