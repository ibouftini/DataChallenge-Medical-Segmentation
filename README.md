# Medical Segmentation of Anatomical Structures with Missing Labels

**Team:** Imade Bouftini, Taha Boukaidi Laghzaoui, Aicha Brequecha, Ilyas Hlaili

**Supervisor:** Claire Brecheteau

**Company:** Radium


---

## Introduction

Computed Tomography (CT) imaging is fundamental to clinical diagnosis and treatment planning, providing detailed
cross-sectional views of internal anatomy. Manual segmentation of anatomical structures in CT scans is time-
consuming and subject to inter-observer variability, necessitating Computer-Aided Detection (CAD) systems for
automated organ delineation. This work addresses a CT multi-organ segmentation challenge featuring 800 partially
annotated images (where visible organs may be unlabeled), 1200 unlabeled images, and 54 anatomical classes
including tumors, evaluated on fully annotated test data using the Dice similarity coefficient.

<p align="center">
  <img src="https://challengedata.ens.fr/media/public/raidium_2024_1.png" alt="Challenge Illustration" width="500"/>
</p>

We developed it as part of the [Data Challenge](https://challengedata.ens.fr/participants/challenges/165/) organized by Data team at ENS Paris Saclay and Data lab at Institut Louis Bachelier.

Our approach combines four deep learning models (DeepLabV3+ and U-Net++ with different encoders) with morphological post-processing. The ensemble reached 59% Dice score on the test set, earning us 3rd place in the competition.




---

## Quick Start

```bash
# Install dependencies
pip install -r 1_requirements.txt

# Train model
python 2_supervised_training.py --arch deeplabv3plus --encoder resnet50 --epochs 50

# Ensemble prediction
python 3_ensemble_predict.py --mode test \
    --models model1.pth,model2.pth \
    --encoders resnet50,resnet101 \
    --architectures deeplabv3plus,unetplusplus \
    --output ./submission

# Visualize results
python 4_visualize.py --submission ./submission/submission.csv --num_images 20
```

---

## Dataset

- **Train**: 800 grayscale images (256x256), 54 masks
- **Test**: 500 grayscale images (256x256)
- **Format**: PNG images + CSV annotations + JSON labels

Download the dataset:
```bash
wget https://challengedata.ens.fr/media/public/train-images.zip
wget https://challengedata.ens.fr/media/public/test-images.zip
wget https://challengedata.ens.fr/media/public/label_Hnl61pT.csv -O y_train.csv
```

The dataset should be placed in the `data/raw/` directory with the following structure:
```
data/raw/
├── train-images/
├── test-images/
├── y_train.csv
└── annotated_labels.json
```

---

## Architecture

**Supported Models:**
- DeepLabV3+ (ResNet50/ResNet101)
- U-Net++ (ResNet50/ResNet101)

**Training:**
- Loss: 0.3 x CrossEntropy + 0.7 x Dice
- Optimizer: AdamW (lr=8e-4, weight_decay=1e-4)
- Batch size: 10 (effective 20 with gradient accumulation)
- Mixed precision (FP16)

---

## Usage

### Training
```bash
# DeepLabV3+ + ResNet50
python 2_supervised_training.py --arch deeplabv3plus --encoder resnet50 --epochs 50

# U-Net++ + ResNet101
python 2_supervised_training.py --arch unetplusplus --encoder resnet101 --epochs 100

# Resume training
python 2_supervised_training.py --resume ./checkpoints/best_model.pth
```

### Ensemble Prediction
```bash
# Test set
python 3_ensemble_predict.py --mode test \
    --models model1.pth,model2.pth \
    --encoders resnet50,resnet101 \
    --architectures deeplabv3plus,unetplusplus \
    --output ./submission

# Validation set
python 3_ensemble_predict.py --mode validation \
    --models model1.pth,model2.pth \
    --encoders resnet50,resnet101 \
    --architectures deeplabv3plus,unetplusplus
```

---

## Files

```
0_readme.md                    # This file
1_requirements.txt             # Dependencies
2_supervised_training.py       # Training script
3_ensemble_predict.py          # Ensemble prediction
4_visualize.py                 # Visualization
Rapport_final.pdf              # Final competition report
visualization_examples/        # Sample prediction visualizations
```
