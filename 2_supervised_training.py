"""
Multi-Architecture Medical Image Segmentation Training Script

Supported Architectures:
    - DeepLabV3+ (deeplabv3plus)
    - U-Net++ (unetplusplus)

Supported Encoders:
    - ResNet50 (resnet50)
    - ResNet101 (resnet101)

Usage:
    # Train DeepLabV3+ with ResNet50 (default)
    python 1_supervised_training.py

    # Train U-Net++ with ResNet101
    python 1_supervised_training.py --arch unetplusplus --encoder resnet101

    # Train DeepLabV3+ with ResNet101
    python 1_supervised_training.py --arch deeplabv3plus --encoder resnet101

    # Resume training from checkpoint
    python 1_supervised_training.py --resume ./checkpoints/best_model.pth

    # Resume with custom learning rate (for fine-tuning)
    python 1_supervised_training.py --resume ./checkpoints/best_model.pth --lr 1e-4

    # Full example
    python 1_supervised_training.py --arch unetplusplus --encoder resnet101 --epochs 200 --lr 8e-4
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.amp as amp
from tqdm import tqdm
import cv2
import albumentations as A
import argparse

# --- NEW REQUIREMENT ---
# Import the library for ResNet encoder
import segmentation_models_pytorch as smp

# ==========================================
# SECTION 1: ARGUMENT PARSER & CONFIGURATION
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Architecture Medical Image Segmentation Training')
    parser.add_argument('--arch', type=str, default='deeplabv3plus',
                        choices=['deeplabv3plus', 'unetplusplus'],
                        help='Architecture: deeplabv3plus or unetplusplus (default: deeplabv3plus)')
    parser.add_argument('--encoder', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'],
                        help='Encoder: resnet50 or resnet101 (default: resnet50)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., ./checkpoints/best_model.pth)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (e.g., 1e-4 for fine-tuning)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 50)')
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()

# ==========================================
# PATHS ADAPTED FOR LOCAL WSL STRUCTURE
# ==========================================
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data/raw")
X_TRAIN_DIR = os.path.join(DATA_DIR, "train-images")  # Adjust to "train-data" if that's your folder name
Y_TRAIN_CSV = os.path.join(DATA_DIR, "y_train.csv")
JSON_PATH = os.path.join(DATA_DIR, "annotated_labels.json")

# Outputs (architecture-specific directories)
ARCH_NAME = args.arch
ENCODER_NAME = args.encoder
OUTPUT_DIR = f"./checkpoints_{ARCH_NAME}_{ENCODER_NAME}"
IMG_SAVE_DIR = f"./training_visuals_{ARCH_NAME}_{ENCODER_NAME}"
LOG_FILE = f"./training_log_{ARCH_NAME}_{ENCODER_NAME}.txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

# Hyperparameters
IMG_SIZE = 256
NUM_CLASSES = 55
BATCH_SIZE = 10  # Adjust based on GPU memory
ACCUMULATION_STEPS = 2  # Effective batch size: 20
LR = args.lr if args.lr is not None else 8e-4  # Default learning rate
MIN_LR = 1e-6
EPOCHS = args.epochs  # Use command-line argument
VAL_SPLIT = 0.2
EARLY_STOP_PATIENCE = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ImageNet normalization for grayscale (average of RGB values)
# ImageNet RGB: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# For grayscale, we use the average
IMAGENET_MEAN = 0.449  # (0.485 + 0.456 + 0.406) / 3
IMAGENET_STD = 0.226   # (0.229 + 0.224 + 0.225) / 3

# CLASS WEIGHTS: Background (Class 0) is weighted 10x less than foreground.
CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
CLASS_WEIGHTS[0] = 0.1
if DEVICE == 'cuda':
    CLASS_WEIGHTS = CLASS_WEIGHTS.cuda()

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"=" * 80)
print(f"MULTI-ARCHITECTURE TRAINING CONFIGURATION")
print(f"=" * 80)
print(f"Architecture: {ARCH_NAME.upper()}")
print(f"Encoder: {ENCODER_NAME.upper()}")
print(f"Device: {DEVICE} (Mixed Precision)")
print(f"Data Directory: {DATA_DIR}")
print(f"Train Images: {X_TRAIN_DIR}")
print(f"Batch Size: {BATCH_SIZE} (Effective: {BATCH_SIZE * ACCUMULATION_STEPS})")
print(f"Learning Rate: {LR}")
print(f"Epochs: {EPOCHS}")
print(f"L2 Regularization (Weight Decay): 1e-4")
print(f"Seed: {SEED}")
print(f"Output Directory: {OUTPUT_DIR}")
if args.resume:
    print(f"Resume from: {args.resume}")
print(f"=" * 80)

# ==========================================
# SECTION 2: MEDICAL-SAFE DATA AUGMENTATION (CLEANED)
# ==========================================
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=0,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.3
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.3
    ),
])

# ==========================================
# SECTION 3: DATA PIPELINE (CLEANED)
# ==========================================
def load_data_into_memory():
    """Loads raw CSV and JSON data"""
    print(f"Loading {Y_TRAIN_CSV} into RAM...")
    try:
        df = pd.read_csv(Y_TRAIN_CSV, index_col=0)
        data_values = df.T.values.reshape((-1, IMG_SIZE, IMG_SIZE))

        with open(JSON_PATH, 'r') as f:
            full_json = json.load(f)

        print("Slicing to first 800 labeled images...")
        return data_values[:800], full_json[:800]

    except Exception as e:
        print(f"ERROR: {e}")
        return None, None

class ChallengeDataset(Dataset):
    def __init__(self, image_dir, labels_array, json_list, transform=None):
        self.image_dir = image_dir
        self.labels = labels_array
        self.json_list = json_list
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load Image
        img_path = os.path.join(self.image_dir, f"{idx}.png")
        if os.path.exists(img_path):
            image = cv2.imread(img_path, 0)
            if image is None:
                image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        else:
            image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # Label & Filter Logic
        raw_label = self.labels[idx].astype(np.int32)
        valid_ids = self.json_list[idx]

        # Filtered Label (used for training loss, setting unannotated regions to -100)
        filtered_label = np.full_like(raw_label, -100, dtype=np.int32)
        filtered_label[raw_label == 0] = 0
        mask = np.isin(raw_label, valid_ids)
        filtered_label[mask] = raw_label[mask]

        # Apply Augmentation (if training)
        if self.transform:
            augmented = self.transform(image=image, mask=filtered_label)
            image = augmented['image']
            filtered_label = augmented['mask']

        # Normalize to [0, 1] then apply ImageNet normalization
        image = image.astype(np.float32) / 255.0
        # Apply ImageNet normalization for better transfer learning
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        # Grayscale single channel (SMP handles pretrained weight adaptation automatically)
        image = np.expand_dims(image, axis=0)  # Shape: (1, H, W)

        # Valid Vector for metric calculation (Excludes Class 0)
        valid_vector = torch.zeros(NUM_CLASSES)
        if len(valid_ids) > 0:
            valid_ids_tensor = torch.tensor(valid_ids, dtype=torch.long)
            valid_vector[valid_ids_tensor] = 1

        return {
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(filtered_label).long(),
            'raw_label': torch.from_numpy(self.labels[idx].astype(np.int32)).long(),
            'valid_vec': valid_vector
        }

# ==========================================
# SECTION 4: MULTI-ARCHITECTURE MODEL CREATION
# ==========================================
def create_model(architecture, encoder_name="resnet50", n_classes=NUM_CLASSES, encoder_weights="imagenet"):
    """
    Creates a segmentation model with specified architecture and encoder.

    Supported Architectures:
    - DeepLabV3+: ASPP + encoder-decoder with skip connections
      * Best for multi-scale objects and varying sizes
      * 25M params (ResNet50), 44M params (ResNet101)

    - U-Net++: Nested U-Net with deep supervision
      * Better feature fusion via nested skip pathways
      * Excellent for medical imaging
      * 31M params (ResNet50), 53M params (ResNet101)

    Supported Encoders:
    - ResNet50: 25M params, good balance of capacity and efficiency
    - ResNet101: 44M params, higher capacity for complex patterns

    SMP automatically handles grayscale to pretrained weight conversion.

    Args:
        architecture: 'deeplabv3plus' or 'unetplusplus'
        encoder_name: 'resnet50' or 'resnet101'
        n_classes: Number of output classes (default: 55)
        encoder_weights: Pretrained weights source (default: 'imagenet')

    Returns:
        PyTorch model
    """
    architecture = architecture.lower()

    if architecture == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=n_classes,
            activation=None,
            encoder_output_stride=16
        )
    elif architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=n_classes,
            activation=None,
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Choose 'deeplabv3plus' or 'unetplusplus'")

    return model

# ==========================================
# SECTION 5: POLYLR SCHEDULER (DEEPLABV3+ ORIGINAL)
# ==========================================
class PolyLR(optim.lr_scheduler._LRScheduler):
    """
    Polynomial Learning Rate Scheduler used in the original DeepLabV3+ paper.

    LR decays polynomially: lr = base_lr * (1 - epoch/max_epochs)^power

    Args:
        optimizer: Optimizer instance
        max_epochs: Total number of training epochs
        power: Polynomial power (0.9 is standard for segmentation)
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_epochs) ** self.power
                for base_lr in self.base_lrs]

# ==========================================
# SECTION 6: COMBINED LOSS (AGGRESSIVE FOR FOREGROUND) - MODIFIED FOR EXPLICIT PER-ORGAN AVERAGING
# ==========================================
class DiceCELoss(nn.Module):
    # Loss weights: CE=0.3, Dice=0.7
    def __init__(self, ignore_index=-100, ce_weight=0.3, dice_weight=0.7, class_weights=None):
        super().__init__()
        # Cross Entropy uses class_weights to de-emphasize background errors
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)

        # Dice Loss (PER-IMAGE, PER-ORGAN - matches evaluation metric)
        # Uses SOFT probabilities for differentiability
        pred_soft = torch.softmax(pred, dim=1)
        batch_size = pred.shape[0]
        num_classes = pred.shape[1]
        
        batch_dice_loss = 0.0
        total_images = 0
        
        # Compute Dice per image, then average
        for i in range(batch_size):
            pred_i = pred_soft[i]  # (C, H, W)
            target_i = target[i]   # (H, W)
            valid_mask_i = (target_i != self.ignore_index)
            
            image_dice_loss = 0.0
            organ_count = 0
            
            # For each organ class present in this image
            for c in range(1, num_classes):  # Skip background (Class 0)
                target_c = (target_i == c).float()  # (H, W)
                
                # Only compute Dice for organs that are PRESENT (target_c.sum() > 0)
                if target_c.sum() > 0:
                    pred_c = pred_i[c]  # (H, W)
                    
                    # Apply valid mask to both pred and target
                    pred_c_valid = pred_c[valid_mask_i]
                    target_c_valid = target_c[valid_mask_i]
                    
                    intersection = (pred_c_valid * target_c_valid).sum()
                    union = pred_c_valid.sum() + target_c_valid.sum()
                    
                    if union > 0:
                        # Soft Dice calculation: (2 * I) / (U)
                        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
                        image_dice_loss += (1.0 - dice)  # Convert to loss
                        organ_count += 1
            
            # Average Dice loss across organs in this image
            if organ_count > 0:
                batch_dice_loss += (image_dice_loss / organ_count)
                total_images += 1
        
        # Average across images
        if total_images > 0:
            dice_loss = batch_dice_loss / total_images
        else:
            dice_loss = torch.tensor(0.0, device=pred.device)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

# ==========================================
# SECTION 7: METRICS AND VISUALS (FIXED FOR PRESENT ORGANS ONLY)
# ==========================================
def compute_batch_dice(preds, targets, valid_vecs):
    """
    Computes per-image Dice, averaging only over organs that are
    PRESENT in the ground truth (t.sum() > 0).
    
    Uses HARD predictions (argmax) for the metric calculation.
    """
    pred_mask = torch.argmax(preds, dim=1)
    batch_dice_sum = 0.0
    images_count = 0

    for i in range(preds.shape[0]):
        img_dice_sum = 0.0
        valid_organs_count = 0

        # valid_classes contains ONLY foreground classes (1-54) for this image
        valid_classes = torch.nonzero(valid_vecs[i]).squeeze(-1)
        if valid_classes.dim() == 0:
            valid_classes = valid_classes.unsqueeze(0)

        for c in valid_classes:
            p = (pred_mask[i] == c).float() # Hard prediction mask
            t = (targets[i] == c).float()   # Hard target mask

            # CRITICAL FIX: Only score classes that are present (t.sum() > 0)
            if t.sum() > 0:
                intersection = (p * t).sum()
                union = p.sum() + t.sum()

                if union > 0:
                    dice = (2. * intersection) / (union + 1e-8)
                    img_dice_sum += dice
                    valid_organs_count += 1
                else:
                    # Organ is present (t.sum() > 0) but prediction is 0. Dice = 0.
                    img_dice_sum += 0.0
                    valid_organs_count += 1
            # else: Organ is absent (t.sum() == 0). Skip.

        if valid_organs_count > 0:
            batch_dice_sum += (img_dice_sum / valid_organs_count)
            images_count += 1

    if images_count == 0:
        return 0.0

    final_dice = batch_dice_sum / images_count
    if isinstance(final_dice, torch.Tensor):
        return final_dice.item()
    return final_dice

def save_visuals(epoch, img, tgt, raw_tgt, pred, prefix="train"):
    """Saves 5-panel visual: Original | Raw GT | Filtered GT | Prediction | Overlay"""

    # Grayscale input is single channel
    i = img[0, 0].cpu().numpy()
    r = raw_tgt[0].cpu().numpy()
    t = tgt[0].cpu().numpy()
    p = torch.argmax(pred[0], dim=0).cpu().numpy()

    # Prepare GT/Prediction for Color Mapping
    t_clean = t.copy()
    t_clean[t_clean == -100] = 0

    # Scale and convert to uint8
    r_vis = (r * 5).clip(0, 255).astype(np.uint8)
    t_vis = (t_clean * 5).clip(0, 255).astype(np.uint8)
    p_vis = (p * 5).clip(0, 255).astype(np.uint8)

    # Apply Color Map (denormalize from ImageNet normalization first)
    i_denorm = (i * IMAGENET_STD + IMAGENET_MEAN) * 255
    i_vis = i_denorm.clip(0, 255).astype(np.uint8)
    i_vis = cv2.cvtColor(i_vis, cv2.COLOR_GRAY2BGR)

    r_c = cv2.applyColorMap(r_vis, cv2.COLORMAP_JET)
    t_c = cv2.applyColorMap(t_vis, cv2.COLORMAP_JET)
    p_c = cv2.applyColorMap(p_vis, cv2.COLORMAP_JET)

    # Calculate Overlay (Prediction on Original Image)
    overlay = cv2.addWeighted(i_vis, 0.6, p_c, 0.4, 0)

    # Stack Images (5 panels)
    res = np.hstack([i_vis, r_c, t_c, p_c, overlay])

    filename = os.path.join(IMG_SAVE_DIR, f"{prefix}_epoch_{epoch:03d}.png")
    cv2.imwrite(filename, res)

def log_metrics(epoch, train_loss, train_dice, val_loss, val_dice, lr):
    """Log metrics to file and console"""
    log_line = f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {lr:.6f}\n"

    with open(LOG_FILE, 'a') as f:
        f.write(log_line)

    print(f"\n{'='*80}")
    print(log_line.strip())
    print(f"{'='*80}\n")

# ==========================================
# SECTION 8: TRAINING LOOP (RESNET MODIFIED WITH POLYLR)
# ==========================================
def main():
    # Load Data
    y_data, json_data = load_data_into_memory()
    if y_data is None:
        return

    # Create datasets
    train_dataset = ChallengeDataset(X_TRAIN_DIR, y_data, json_data, transform=train_transform)
    val_dataset = ChallengeDataset(X_TRAIN_DIR, y_data, json_data, transform=None)

    # Split indices
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VAL_SPLIT * dataset_size))
    np.random.seed(SEED)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                              pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
                            pin_memory=True, num_workers=4)

    print(f"Train: {len(train_indices)} | Val: {len(val_indices)}")

    # Model Setup (architecture and encoder from command-line arguments)
    print(f"\n[MODEL] Creating {ARCH_NAME.upper()} with {ENCODER_NAME.upper()} encoder...")
    model = create_model(
        architecture=ARCH_NAME,
        encoder_name=ENCODER_NAME,
        n_classes=NUM_CLASSES,
        encoder_weights="imagenet"
    ).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters: {total_params:,}")
    print(f"[MODEL] Trainable parameters: {trainable_params:,}")

    # Optimizer (L2 Regularization: 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Criterion (CE=0.3, Dice=0.7, Class Weights applied)
    criterion = DiceCELoss(ignore_index=-100, ce_weight=0.3, dice_weight=0.7,
                           class_weights=CLASS_WEIGHTS)

    # Learning Rate Scheduler (PolyLR - DeepLabV3+ Original)
    scheduler = PolyLR(optimizer, max_epochs=EPOCHS, power=0.9)

    scaler = torch.amp.GradScaler(DEVICE)

    best_val_dice = 0.0
    patience_counter = 0
    start_epoch = 1

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\n[OK] Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=DEVICE)

            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            print("[OK] Model weights loaded")

            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("[OK] Optimizer state loaded")

            # Override learning rate if specified
            if args.lr is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                print(f"[OK] Learning rate overridden to: {args.lr}")

            # Load training state
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_dice = checkpoint.get('val_dice', 0.0)

            print(f"[OK] Resuming from epoch {start_epoch}")
            print(f"[OK] Best validation Dice so far: {best_val_dice:.4f}\n")
        else:
            print(f"[WARNING] Checkpoint file not found: {args.resume}")
            print("Starting training from scratch...\n")

    # Initialize log file
    mode = 'a' if args.resume and os.path.exists(LOG_FILE) else 'w'
    with open(LOG_FILE, mode) as f:
        if mode == 'w':
            f.write("Training Log\n")
            f.write("="*80 + "\n")
        else:
            f.write(f"\n{'='*80}\n")
            f.write(f"RESUMED TRAINING - Epoch {start_epoch}\n")
            f.write(f"{'='*80}\n")

    print(f"\n[START] Training {ARCH_NAME.upper()} with {ENCODER_NAME.upper()} (Gradient Accumulation)...\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        # TRAINING PHASE
        model.train()
        train_loss = 0
        train_dice = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]", leave=True)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            img = batch['image'].to(DEVICE, non_blocking=True)
            tgt = batch['label'].to(DEVICE, non_blocking=True)
            valid = batch['valid_vec'].to(DEVICE, non_blocking=True)

            with torch.amp.autocast(DEVICE):
                preds = model(img)
                loss = criterion(preds, tgt) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            # Gradient Accumulation
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUMULATION_STEPS
            # NOTE: compute_batch_dice uses the RAW_LABEL (target) for PRESENT check
            dice = compute_batch_dice(preds.detach(), batch['raw_label'].to(DEVICE), valid)
            train_dice += dice

            pbar.set_postfix({'Loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}",
                              'Dice': f"{dice:.4f}"})

        # FIX: Ensure final update for partial accumulation batch
        if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        # VALIDATION PHASE
        model.eval()
        val_loss = 0
        val_dice = 0

        last_img, last_tgt, last_raw_tgt, last_preds, last_valid = None, None, None, None, None

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [VAL]", leave=True)
            for batch in pbar_val:
                img = batch['image'].to(DEVICE, non_blocking=True)
                tgt = batch['label'].to(DEVICE, non_blocking=True)
                valid = batch['valid_vec'].to(DEVICE, non_blocking=True)
                raw_tgt = batch['raw_label'].to(DEVICE, non_blocking=True)

                with torch.amp.autocast(DEVICE):
                    preds = model(img)
                    loss = criterion(preds, tgt)

                val_loss += loss.item()
                # NOTE: compute_batch_dice uses the RAW_LABEL (target) for PRESENT check
                dice = compute_batch_dice(preds, raw_tgt, valid)
                val_dice += dice

                pbar_val.set_postfix({'Loss': f"{loss.item():.4f}",
                                      'Dice': f"{dice:.4f}"})

                last_img, last_tgt, last_raw_tgt, last_preds, last_valid = img, tgt, raw_tgt, preds, valid

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        log_metrics(epoch, avg_train_loss, avg_train_dice, avg_val_loss, avg_val_dice, current_lr)

        # Save visuals (using the stored last batch)
        save_visuals(epoch, last_img, last_tgt, last_raw_tgt, last_preds, prefix="val")

        # Learning Rate Scheduling (PolyLR steps every epoch)
        scheduler.step()

        # Save best model logic
        if avg_val_dice > best_val_dice:
            improvement = avg_val_dice - best_val_dice
            print(f"[NEW BEST] Val Dice: {avg_val_dice:.4f} (+{improvement:.4f})")
            best_val_dice = avg_val_dice
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'train_dice': avg_train_dice,
            }, os.path.join(OUTPUT_DIR, "best_model.pth"))
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        # Early Stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[STOP] Early stopping triggered at epoch {epoch}")
            break

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': avg_val_dice,
            }, os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch}.pth"))

    # Save final model after training completes
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))

    print(f"\n[COMPLETE] Training Complete! Best Val Dice: {best_val_dice:.4f}")
    print(f"[INFO] Full log saved to: {LOG_FILE}")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()