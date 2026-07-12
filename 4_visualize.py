"""
Visualization Tool for Medical Image Segmentation
Visualizes predictions with original images and overlays.

Usage:
    python 4_visualize.py --submission ./submission/submission.csv --num_images 20
    python 4_visualize.py --submission ./submission/submission.csv --show_grid
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
parser = argparse.ArgumentParser(description='Visualize predictions')
parser.add_argument('--submission', type=str, default='./submission/submission.csv',
                    help='Path to submission CSV file (default: ./submission/submission.csv)')
parser.add_argument('--test_dir', type=str, default='data/raw/test-images',
                    help='Path to test images directory')
parser.add_argument('--output_dir', type=str, default='./visualizations',
                    help='Output directory for visualizations')
parser.add_argument('--start_idx', type=int, default=0,
                    help='Starting image index')
parser.add_argument('--num_images', type=int, default=20,
                    help='Number of images to visualize')
parser.add_argument('--show_grid', action='store_true',
                    help='Display images in a grid instead of saving individual files')
args = parser.parse_args()

IMG_SIZE = 256
NUM_CLASSES = 55

print("="*80)
print("PREDICTION VISUALIZATION")
print("="*80)
print(f"Submission file: {args.submission}")
print(f"Test directory: {args.test_dir}")
print(f"Output directory: {args.output_dir}")
print(f"Images to visualize: {args.num_images} (starting from index {args.start_idx})")
print("="*80)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# ==========================================
# LOAD SUBMISSION
# ==========================================
print("\nLoading submission CSV...")
df = pd.read_csv(args.submission, index_col=0)

# Get column names (image filenames)
image_columns = df.columns.tolist()
print(f"Loaded {len(image_columns)} predictions")

# Convert to numpy array
# Shape: (num_pixels, num_images)
predictions_flat = df.values
print(f"Data shape: {predictions_flat.shape}")

# Reshape predictions to (num_images, IMG_SIZE, IMG_SIZE)
num_images_total = predictions_flat.shape[1]
predictions = predictions_flat.T.reshape(num_images_total, IMG_SIZE, IMG_SIZE)
print(f"Reshaped to: {predictions.shape}")

# ==========================================
# COLOR PALETTE
# ==========================================
def generate_distinct_colors(num_classes):
    """
    Generate distinct colors for each class using multiple color schemes
    Returns a lookup table (LUT) for 55 classes with RGB colors
    """
    colors = []

    # Tab20 colormap (20 distinct colors)
    tab20 = plt.cm.get_cmap('tab20', 20)
    for i in range(20):
        rgb = tab20(i)[:3]
        colors.append([int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255)])  # BGR for OpenCV

    # Set3 colormap (12 distinct colors)
    set3 = plt.cm.get_cmap('Set3', 12)
    for i in range(12):
        rgb = set3(i)[:3]
        colors.append([int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255)])

    # Paired colormap (12 distinct colors)
    paired = plt.cm.get_cmap('Paired', 12)
    for i in range(12):
        rgb = paired(i)[:3]
        colors.append([int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255)])

    # Accent colormap (8 distinct colors)
    accent = plt.cm.get_cmap('Accent', 8)
    for i in range(8):
        rgb = accent(i)[:3]
        colors.append([int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255)])

    # Dark2 colormap (8 distinct colors)
    dark2 = plt.cm.get_cmap('Dark2', 8)
    for i in range(3):  # Only take 3 to reach 55 total
        rgb = dark2(i)[:3]
        colors.append([int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255)])

    # Ensure we have exactly num_classes colors
    colors = colors[:num_classes]

    # Create LUT (Look-Up Table) - 256x3 array
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[0] = [0, 0, 0]  # Background is black
    for i, color in enumerate(colors):
        lut[i] = color

    return lut

# Generate color LUT once
COLOR_LUT = generate_distinct_colors(NUM_CLASSES)

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================
def load_test_image(idx, test_dir):
    """Load test image from directory"""
    img_path = os.path.join(test_dir, f"{idx}.png")
    if os.path.exists(img_path):
        image = cv2.imread(img_path, 0)  # Grayscale
        if image is None:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        return image
    else:
        print(f"Warning: Image {idx}.png not found, using blank image")
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

def create_visualization(original, prediction, idx, save_path=None):
    """
    Create 3-panel visualization: Original | Prediction | Overlay

    Args:
        original: Original grayscale image (H, W)
        prediction: Prediction mask with class labels (H, W)
        idx: Image index
        save_path: Path to save image (if None, returns the image)

    Returns:
        Combined visualization image if save_path is None
    """
    # Convert original to BGR for overlay
    orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # Colorize prediction mask using distinct color palette
    # Apply LUT directly to get BGR colors for each pixel
    pred_color = COLOR_LUT[prediction.astype(np.uint8)]

    # Create overlay
    overlay = cv2.addWeighted(orig_bgr, 0.6, pred_color, 0.4, 0)

    # Stack horizontally: Original | Prediction | Overlay
    result = np.hstack([orig_bgr, pred_color, overlay])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (255, 255, 255)

    cv2.putText(result, f"Original (Image {idx})", (10, 20), font, font_scale, font_color, font_thickness)
    cv2.putText(result, "Prediction", (IMG_SIZE + 10, 20), font, font_scale, font_color, font_thickness)
    cv2.putText(result, "Overlay", (2 * IMG_SIZE + 10, 20), font, font_scale, font_color, font_thickness)

    # Add statistics
    unique_classes = np.unique(prediction)
    stats_text = f"Classes: {len(unique_classes)}"
    cv2.putText(result, stats_text, (10, IMG_SIZE - 10), font, 0.4, font_color, 1)

    if save_path:
        cv2.imwrite(save_path, result)
        return None
    else:
        return result

def create_grid_visualization(images_data, grid_cols=4):
    """
    Create a grid visualization of multiple images

    Args:
        images_data: List of (original, prediction, idx) tuples
        grid_cols: Number of columns in the grid

    Returns:
        Grid visualization
    """
    n = len(images_data)
    grid_rows = (n + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 6, grid_rows * 2))

    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1:
        axes = axes.reshape(1, -1)
    elif grid_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, (original, prediction, idx) in enumerate(images_data):
        row = i // grid_cols
        col = i % grid_cols

        # Create 3-panel visualization
        vis = create_visualization(original, prediction, idx, save_path=None)

        # Convert BGR to RGB for matplotlib
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        axes[row, col].imshow(vis_rgb)
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Image {idx}", fontsize=10, fontweight='bold')

    # Hide empty subplots
    for i in range(n, grid_rows * grid_cols):
        row = i // grid_cols
        col = i % grid_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    return fig

# ==========================================
# MAIN VISUALIZATION
# ==========================================
print(f"\nGenerating visualizations...")

# Extract image indices from column names (e.g., "0.png" -> 0)
image_indices = []
for col in image_columns:
    idx_str = col.replace('.png', '')
    try:
        idx = int(idx_str)
        image_indices.append(idx)
    except ValueError:
        print(f"Warning: Could not parse index from column: {col}")

# Determine which images to visualize
end_idx = min(args.start_idx + args.num_images, len(image_indices))
selected_indices = image_indices[args.start_idx:end_idx]

print(f"Visualizing {len(selected_indices)} images...")

if args.show_grid:
    # Grid visualization mode
    images_data = []

    for i, img_idx in enumerate(tqdm(selected_indices, desc="Loading images")):
        # Load original image
        original = load_test_image(img_idx, args.test_dir)

        # Get prediction
        prediction = predictions[args.start_idx + i]

        images_data.append((original, prediction, img_idx))

    # Create grid
    fig = create_grid_visualization(images_data, grid_cols=4)

    # Save grid
    grid_path = os.path.join(args.output_dir, f"grid_{args.start_idx}_to_{end_idx-1}.png")
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nGrid visualization saved to: {grid_path}")

else:
    # Individual file mode
    for i, img_idx in enumerate(tqdm(selected_indices, desc="Generating visualizations")):
        # Load original image
        original = load_test_image(img_idx, args.test_dir)

        # Get prediction
        prediction = predictions[args.start_idx + i]

        # Create and save visualization
        save_path = os.path.join(args.output_dir, f"test_{img_idx}.png")
        create_visualization(original, prediction, img_idx, save_path=save_path)

    print(f"\nVisualizations saved to: {args.output_dir}/")

# ==========================================
# STATISTICS
# ==========================================
print("\n" + "="*80)
print("STATISTICS")
print("="*80)

all_classes = []
for i in range(len(selected_indices)):
    pred = predictions[args.start_idx + i]
    all_classes.extend(np.unique(pred))

unique_classes = sorted(set(all_classes))

print(f"Images visualized: {len(selected_indices)}")
print(f"Unique classes found: {len(unique_classes)}")
print(f"Class range: {min(unique_classes)} - {max(unique_classes)}")

# Class distribution
print(f"\nClass distribution (top 10):")
class_counts = {}
for cls in all_classes:
    class_counts[cls] = class_counts.get(cls, 0) + 1

sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
for cls, count in sorted_classes[:10]:
    print(f"  Class {cls:2d}: {count:8d} pixels")

print("="*80)
print("DONE!")
print("="*80)
