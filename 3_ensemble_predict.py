"""
Multi-Model Ensemble Prediction Script

Usage:

    # 3 DeepLabV3+ + 1 U-Net++
    python ensemble_predict.py --mode test \
        --models ./checkpoints/model1.pth,./checkpoints/model2.pth,./checkpoints/model3.pth,./checkpoints_unetpp/best_model.pth \
        --encoders resnet50,resnet50,resnet101,resnet101 \
        --architectures deeplabv3plus,deeplabv3plus,deeplabv3plus,unetplusplus \
        --tta \
        --output ./submission

    # Optional flags:
    --tta              Enable Test-Time Augmentation (horizontal flip ONLY - may reduce performance!)
    --weights 0.25,0.25,0.25,0.25  Custom ensemble weights (default: equal)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import zipfile

# ==========================================
# ARGUMENT PARSER
# ==========================================
parser = argparse.ArgumentParser(description='Ensemble prediction with multiple models and architectures')
parser.add_argument('--mode', type=str, required=True, choices=['validation', 'test'],
                    help='Mode: validation (evaluate on val set) or test (generate submission)')
parser.add_argument('--models', type=str, required=True,
                    help='Comma-separated paths to model checkpoints (e.g., "model1.pth,model2.pth,model3.pth")')
parser.add_argument('--encoders', type=str, required=True,
                    help='Comma-separated encoder names (e.g., "resnet50,resnet50,resnet101")')
parser.add_argument('--architectures', type=str, required=True,
                    help='Comma-separated architecture names (e.g., "deeplabv3plus,deeplabv3plus,unetplusplus")')
parser.add_argument('--weights', type=str, default=None,
                    help='Ensemble weights as comma-separated (e.g., "0.3,0.3,0.4")')
parser.add_argument('--tta', action='store_true', help='Enable Test-Time Augmentation')
parser.add_argument('--output', type=str, default='./submission', help='Output directory')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
parser.add_argument('--num_vis', type=int, default=20, help='Number of visualizations')
args = parser.parse_args()

# Parse model paths, encoders, and architectures
MODEL_PATHS = [p.strip() for p in args.models.split(',')]
ENCODERS = [e.strip() for e in args.encoders.split(',')]
ARCHITECTURES = [a.strip() for a in args.architectures.split(',')]

NUM_MODELS = len(MODEL_PATHS)

# Validate inputs
assert len(ENCODERS) == NUM_MODELS, f"Number of encoders ({len(ENCODERS)}) must match number of models ({NUM_MODELS})"
assert len(ARCHITECTURES) == NUM_MODELS, f"Number of architectures ({len(ARCHITECTURES)}) must match number of models ({NUM_MODELS})"

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = "."
X_TRAIN_DIR = os.path.join(BASE_DIR, "data/raw/train-images")
X_TEST_DIR = os.path.join(BASE_DIR, "data/raw/test-images")
Y_TRAIN_CSV = os.path.join(BASE_DIR, "data/raw/y_train.csv")
JSON_PATH = os.path.join(BASE_DIR, "data/raw/annotated_labels.json")

OUTPUT_DIR = args.output
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

IMG_SIZE = 256
NUM_CLASSES = 55
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
VAL_SPLIT = 0.2

IMAGENET_MEAN = 0.449
IMAGENET_STD = 0.226

torch.manual_seed(SEED)
np.random.seed(SEED)

# Parse ensemble weights
if args.weights:
    ENSEMBLE_WEIGHTS = [float(w) for w in args.weights.split(',')]
    assert len(ENSEMBLE_WEIGHTS) == NUM_MODELS, f"Must provide exactly {NUM_MODELS} weights"
    assert abs(sum(ENSEMBLE_WEIGHTS) - 1.0) < 1e-6, "Weights must sum to 1.0"
else:
    ENSEMBLE_WEIGHTS = [1/NUM_MODELS] * NUM_MODELS  # Equal weights

print(f"Device: {DEVICE}")
print(f"Number of Models: {NUM_MODELS}")
print(f"\nModel Details:")
for i, (path, arch, enc) in enumerate(zip(MODEL_PATHS, ARCHITECTURES, ENCODERS), 1):
    print(f"  [{i}] {arch:15s} | {enc:12s} | {path}")
print(f"\nEnsemble Weights: {ENSEMBLE_WEIGHTS}")
if args.tta:
    print(f"TTA: Enabled (Horizontal Flip Only)")
else:
    print(f"TTA: Disabled")
print(f"Output: {OUTPUT_DIR}")
print()

# ==========================================
# DATASETS
# ==========================================
class ValidationDataset(Dataset):
    """Validation dataset with ground truth labels"""
    def __init__(self, image_dir, labels_array, json_list):
        self.image_dir = image_dir
        self.labels = labels_array
        self.json_list = json_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"{idx}.png")
        if os.path.exists(img_path):
            image = cv2.imread(img_path, 0)
            if image is None:
                image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        else:
            image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        original = image.copy()
        raw_label = self.labels[idx].astype(np.int32)
        valid_ids = self.json_list[idx]

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.expand_dims(image, axis=0)

        # Valid vector
        valid_vector = torch.zeros(NUM_CLASSES)
        if len(valid_ids) > 0:
            valid_ids_tensor = torch.tensor(valid_ids, dtype=torch.long)
            valid_vector[valid_ids_tensor] = 1

        return {
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(raw_label).long(),
            'valid_vec': valid_vector,
            'original': original,
            'idx': idx
        }

class TestDataset(Dataset):
    """Test dataset without labels"""
    def __init__(self, image_dir, start_idx=0, end_idx=500):
        self.image_dir = image_dir
        self.indices = []
        for idx in range(start_idx, end_idx):
            img_path = os.path.join(image_dir, f"{idx}.png")
            if os.path.exists(img_path):
                self.indices.append(idx)

        print(f"Found {len(self.indices)} test images")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx]
        img_path = os.path.join(self.image_dir, f"{img_idx}.png")

        image = cv2.imread(img_path, 0)
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        original = image.copy()

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.expand_dims(image, axis=0)

        return {
            'image': torch.from_numpy(image),
            'original': original,
            'idx': img_idx
        }

# ==========================================
# MODEL LOADING
# ==========================================
def create_model(architecture, encoder_name):
    """Create model based on architecture type"""
    architecture = architecture.lower()

    if architecture == 'deeplabv3plus':
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=1,
            classes=NUM_CLASSES,
            activation=None,
            encoder_output_stride=16
        )
    elif architecture == 'unetplusplus' or architecture == 'unet++':
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=1,
            classes=NUM_CLASSES,
            activation=None,
        )
    elif architecture == 'fpn':
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=1,
            classes=NUM_CLASSES,
            activation=None,
        )
    elif architecture == 'manet':
        return smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=1,
            classes=NUM_CLASSES,
            activation=None,
        )
    elif architecture == 'pan':
        return smp.PAN(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=1,
            classes=NUM_CLASSES,
            activation=None,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

def load_model(checkpoint_path, architecture, encoder_name):
    """Load model with specified architecture and encoder"""
    try:
        model = create_model(architecture, encoder_name).to(DEVICE)

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            val_dice = checkpoint.get('val_dice', 'N/A')
            print(f" Loaded {architecture:15s} | {encoder_name:12s} | Val Dice: {val_dice}")
        else:
            model.load_state_dict(checkpoint)
            print(f" Loaded {architecture:15s} | {encoder_name:12s}")

        model.eval()
        return model
    except Exception as e:
        print(f" Failed to load {checkpoint_path}: {e}")
        return None

# ==========================================
# ENSEMBLE PREDICTION WITH TTA
# ==========================================
def predict_with_tta(models, images, weights):
    """
    Ensemble prediction with Test-Time Augmentation (Advanced)

    TTA includes:
    - Original
    - Horizontal flip

    Args:
        models: List of models
        images: Batch of images (B, 1, H, W)
        weights: Ensemble weights

    Returns:
        Ensembled logits (B, NUM_CLASSES, H, W)
    """
    with torch.no_grad():
        all_ensemble_preds = []

        # Original predictions
        preds_original = []
        for model in models:
            with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                pred = model(images)
            preds_original.append(pred)

        # Weighted ensemble of original
        ensemble_original = sum(w * p for w, p in zip(weights, preds_original))
        all_ensemble_preds.append(ensemble_original)

        if not args.tta:
            return ensemble_original

        # TTA 1: Horizontal flip
        images_h_flip = torch.flip(images, dims=[-1])  # Flip width
        preds_h_flip = []
        for model in models:
            with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                pred = model(images_h_flip)
            pred = torch.flip(pred, dims=[-1])  # Flip back
            preds_h_flip.append(pred)

        ensemble_h_flip = sum(w * p for w, p in zip(weights, preds_h_flip))
        all_ensemble_preds.append(ensemble_h_flip)

        # NOTE: Vertical flip and other TTA strategies were tested and found to HURT
        # performance significantly. Only H-flip is kept as optional, but even it
        # slightly reduces Dice score. Recommend using --tta flag sparingly.

        # Average original + horizontal flip only
        return torch.stack(all_ensemble_preds).mean(dim=0)

# ==========================================
# DICE METRIC
# ==========================================
def compute_batch_dice(preds, targets, valid_vecs):
    """
    Compute Dice score only for present organs
    """
    pred_mask = torch.argmax(preds, dim=1)
    batch_dice_sum = 0.0
    images_count = 0

    for i in range(preds.shape[0]):
        img_dice_sum = 0.0
        valid_organs_count = 0

        valid_classes = torch.nonzero(valid_vecs[i]).squeeze(-1)
        if valid_classes.dim() == 0:
            valid_classes = valid_classes.unsqueeze(0)

        for c in valid_classes:
            p = (pred_mask[i] == c).float()
            t = (targets[i] == c).float()

            if t.sum() > 0:
                intersection = (p * t).sum()
                union = p.sum() + t.sum()

                if union > 0:
                    dice = (2. * intersection) / (union + 1e-8)
                    img_dice_sum += dice
                    valid_organs_count += 1
                else:
                    img_dice_sum += 0.0
                    valid_organs_count += 1

        if valid_organs_count > 0:
            batch_dice_sum += (img_dice_sum / valid_organs_count)
            images_count += 1

    if images_count == 0:
        return 0.0

    final_dice = batch_dice_sum / images_count
    return final_dice.item() if isinstance(final_dice, torch.Tensor) else final_dice

# ==========================================
# VALIDATION MODE
# ==========================================
def validation_mode(models):
    """
    Run validation:
    1. Evaluate each model individually
    2. Evaluate ensemble
    3. Generate comparison visualizations
    """
    print("\n[VALIDATION MODE]")
    print("="*80)

    # Load validation data
    print("Loading validation data...")
    df = pd.read_csv(Y_TRAIN_CSV, index_col=0)
    data_values = df.T.values.reshape((-1, IMG_SIZE, IMG_SIZE))

    with open(JSON_PATH, 'r') as f:
        full_json = json.load(f)

    y_data, json_data = data_values[:800], full_json[:800]

    # Create validation split
    dataset = ValidationDataset(X_TRAIN_DIR, y_data, json_data)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VAL_SPLIT * dataset_size))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    val_indices = indices[:split]

    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler,
                           pin_memory=True, num_workers=4)

    print(f"Validation set: {len(val_indices)} images\n")

    # Evaluate each model individually
    print("Evaluating individual models...")
    model_dices = []

    for i, (model, arch, enc) in enumerate(zip(models, ARCHITECTURES, ENCODERS), 1):
        total_dice = 0
        count = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Model {i} ({arch}/{enc})", leave=False):
                images = batch['image'].to(DEVICE)
                targets = batch['label'].to(DEVICE)
                valid = batch['valid_vec'].to(DEVICE)

                with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                    preds = model(images)

                dice = compute_batch_dice(preds, targets, valid)
                total_dice += dice
                count += 1

        avg_dice = total_dice / count
        model_dices.append(avg_dice)
        print(f"  Model {i} ({arch:15s} / {enc:12s}): Dice = {avg_dice:.4f}")

    # Evaluate ensemble
    print("\nEvaluating ensemble...")
    ensemble_dice = 0
    count = 0

    # Storage for visualizations
    vis_data = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Ensemble", leave=False):
            images = batch['image'].to(DEVICE)
            targets = batch['label'].to(DEVICE)
            valid = batch['valid_vec'].to(DEVICE)
            originals = batch['original'].cpu().numpy()
            indices = batch['idx'].cpu().numpy()

            # Ensemble prediction
            preds = predict_with_tta(models, images, ENSEMBLE_WEIGHTS)

            dice = compute_batch_dice(preds, targets, valid)
            ensemble_dice += dice
            count += 1

            # Store for visualization
            if len(vis_data) < args.num_vis:
                pred_masks = torch.argmax(preds, dim=1).cpu().numpy()
                target_masks = targets.cpu().numpy()

                for i in range(len(indices)):
                    if len(vis_data) >= args.num_vis:
                        break
                    vis_data.append({
                        'original': originals[i],
                        'target': target_masks[i],
                        'pred': pred_masks[i],
                        'idx': indices[i]
                    })

    avg_ensemble_dice = ensemble_dice / count

    # Print results
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    for i, (dice, arch, enc) in enumerate(zip(model_dices, ARCHITECTURES, ENCODERS), 1):
        print(f"Model {i} ({arch:15s} / {enc:12s}): Dice = {dice:.4f}")
    print(f"-"*80)
    print(f"Ensemble ({NUM_MODELS} models):    Dice = {avg_ensemble_dice:.4f}")
    print(f"Improvement:              +{avg_ensemble_dice - max(model_dices):.4f} over best single model")
    print(f"TTA Status:               {'Enabled' if args.tta else 'Disabled'}")
    print("="*80)

    # Generate visualizations
    print(f"\nGenerating {len(vis_data)} comparison visualizations...")
    for data in tqdm(vis_data):
        save_validation_visual(data['original'], data['target'], data['pred'],
                              data['idx'], VIS_DIR)

    # Plot Dice comparison
    plot_dice_comparison(model_dices, avg_ensemble_dice, OUTPUT_DIR)

    print(f"\n Validation complete!")
    print(f"   Visualizations: {VIS_DIR}/")
    print(f"   Dice plot: {OUTPUT_DIR}/dice_comparison.png")

# ==========================================
# TEST MODE
# ==========================================
def test_mode(models):
    """
    Run test prediction:
    1. Generate ensemble predictions on test set
    2. Create submission.csv
    3. Generate visualizations
    """
    print("\n[TEST MODE]")
    print("="*80)

    # Create dataset
    dataset = TestDataset(X_TEST_DIR, start_idx=0, end_idx=500)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                       pin_memory=True, num_workers=4)

    print(f"Test set: {len(dataset)} images\n")

    # Run predictions
    print("Running ensemble predictions...")
    all_predictions = {}
    vis_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            images = batch['image'].to(DEVICE)
            originals = batch['original'].cpu().numpy()
            indices = batch['idx'].cpu().numpy()

            # Ensemble prediction with TTA
            preds = predict_with_tta(models, images, ENSEMBLE_WEIGHTS)
            predictions = torch.argmax(preds, dim=1).cpu().numpy()

            # Store predictions
            for i in range(len(indices)):
                idx = indices[i]
                pred = predictions[i]
                all_predictions[idx] = pred

                # Save visualization
                if vis_count < args.num_vis:
                    save_test_visual(originals[i], pred, idx, VIS_DIR)
                    vis_count += 1

    print(f" Predictions complete! ({vis_count} visualizations)")

    # Create submission CSV
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    create_submission_csv(all_predictions, submission_path)

    # Print statistics
    print_statistics(all_predictions)

    # Zip results
    zip_path = zip_results(OUTPUT_DIR)

    print(f"\n{'='*80}")
    print(f" TEST MODE COMPLETE!")
    print(f"{'='*80}")
    print(f"Submission: {submission_path}")
    print(f"Visualizations: {VIS_DIR}/")
    print(f"Zip: {zip_path}")
    print(f"{'='*80}\n")

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================
def save_validation_visual(original, target, pred, idx, folder):
    """4-panel: Original | Ground Truth | Prediction | Overlay"""
    # Convert to BGR
    orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # Colorize masks
    tgt_vis = (target.astype(float) / NUM_CLASSES * 255).astype(np.uint8)
    pred_vis = (pred.astype(float) / NUM_CLASSES * 255).astype(np.uint8)

    tgt_color = cv2.applyColorMap(tgt_vis, cv2.COLORMAP_JET)
    pred_color = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(orig_bgr, 0.6, pred_color, 0.4, 0)

    # Stack
    result = np.hstack([orig_bgr, tgt_color, pred_color, overlay])

    # Add labels
    cv2.putText(result, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(result, "Ground Truth", (IMG_SIZE+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(result, "Prediction", (2*IMG_SIZE+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(result, "Overlay", (3*IMG_SIZE+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imwrite(os.path.join(folder, f"val_{idx}.png"), result)

def save_test_visual(original, pred, idx, folder):
    """3-panel: Original | Prediction | Overlay"""
    orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    pred_vis = (pred.astype(float) / NUM_CLASSES * 255).astype(np.uint8)
    pred_color = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(orig_bgr, 0.6, pred_color, 0.4, 0)

    result = np.hstack([orig_bgr, pred_color, overlay])

    cv2.putText(result, f"Image {idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(result, "Prediction", (IMG_SIZE+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(result, "Overlay", (2*IMG_SIZE+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imwrite(os.path.join(folder, f"test_{idx}.png"), result)

def plot_dice_comparison(model_dices, ensemble_dice, output_dir):
    """Plot Dice score comparison"""
    fig_width = max(10, NUM_MODELS * 1.5)
    plt.figure(figsize=(fig_width, 6))

    # Create labels with architecture info
    labels = []
    for i, (arch, enc) in enumerate(zip(ARCHITECTURES, ENCODERS), 1):
        arch_short = arch.replace('deeplabv3plus', 'DLv3+').replace('unetplusplus', 'U-Net++')
        enc_short = enc.replace('resnet', 'R')
        labels.append(f'Model {i}\n({arch_short}/{enc_short})')
    labels.append('Ensemble')

    scores = model_dices + [ensemble_dice]

    # Color palette
    colors = ['#3498db'] * NUM_MODELS + ['#2ecc71']

    bars = plt.bar(labels, scores, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylabel('Dice Score', fontsize=12, fontweight='bold')
    plt.title(f'{NUM_MODELS}-Model Ensemble - Validation Dice Scores', fontsize=14, fontweight='bold')
    plt.ylim(min(scores) - 0.02, max(scores) + 0.02)
    plt.grid(axis='y', alpha=0.3)

    # Highlight improvement
    improvement = ensemble_dice - max(model_dices)
    plt.text(len(labels)-1, ensemble_dice - 0.01, f'+{improvement:.4f}', ha='center',
             fontsize=10, color='green', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_comparison.png'), dpi=150)
    plt.close()

# ==========================================
# SUBMISSION CSV
# ==========================================
def create_submission_csv(predictions, output_path):
    """Create submission.csv in required format"""
    print(f"\nCreating submission.csv...")

    sorted_indices = sorted(predictions.keys())

    data = []
    for idx in tqdm(sorted_indices, desc="Formatting"):
        pred = predictions[idx]
        data.append(pred.flatten())

    data_array = np.array(data).T
    column_names = [f"{idx}.png" for idx in sorted_indices]

    df = pd.DataFrame(data_array, columns=column_names)
    df.index = [f"Pixel {i}" for i in range(len(df))]

    df.to_csv(output_path)
    print(f" Submission saved: {output_path}")
    print(f"   Shape: {df.shape}")

def print_statistics(predictions):
    """Print prediction statistics"""
    print(f"\n Prediction Statistics")
    print("=" * 60)

    all_classes = []
    for pred in predictions.values():
        all_classes.extend(np.unique(pred))

    unique_classes = sorted(set(all_classes))

    print(f"Total images: {len(predictions)}")
    print(f"Unique classes: {len(unique_classes)}")
    print(f"Class range: {min(unique_classes)} - {max(unique_classes)}")
    print("=" * 60)

def zip_results(output_dir):
    """Zip submission and visualizations"""
    zip_path = "submission_ensemble.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        submission_path = os.path.join(output_dir, "submission.csv")
        if os.path.exists(submission_path):
            zipf.write(submission_path, "submission.csv")

        vis_dir = os.path.join(output_dir, "visualizations")
        if os.path.exists(vis_dir):
            for filename in os.listdir(vis_dir):
                if filename.endswith('.png'):
                    file_path = os.path.join(vis_dir, filename)
                    zipf.write(file_path, f"visualizations/{filename}")

    return zip_path

# ==========================================
# MAIN
# ==========================================
def main():
    # Load models
    print("\n" + "="*80)
    print("Loading models...")
    print("="*80)

    models = []
    for path, arch, enc in zip(MODEL_PATHS, ARCHITECTURES, ENCODERS):
        model = load_model(path, arch, enc)
        if model is None:
            print(" Failed to load one or more models. Exiting.")
            return
        models.append(model)

    print(f"\n Successfully loaded {NUM_MODELS} models!")
    print("="*80 + "\n")

    # Run mode
    if args.mode == 'validation':
        validation_mode(models)
    else:
        test_mode(models)

if __name__ == "__main__":
    main()