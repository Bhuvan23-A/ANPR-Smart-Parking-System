"""
Script to generate the comprehensive, interactive Jupyter Notebook:
ANPR_Smart_Parking_System.ipynb
"""

import json
import os

def build_notebook():
    cells = []

    def md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Title & Metadata
    md("""
# COURSE: PATTERN RECOGNITION
## CASE STUDY 13: AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) FOR SMART PARKING

**Evaluation Rubric Alignment (100 Marks):**
1. **Problem Identification and Understanding** (10 Marks)
2. **Dataset Collection and Preparation** (10 Marks)
3. **Selection of Suitable Algorithms** (15 Marks)
4. **Algorithm Implementation** (20 Marks)
5. **Results and Performance Analysis** (15 Marks)
6. **Real-world Solution and Interpretation** (20 Marks)
7. **Presentation and Viva Voce Preparation** (10 Marks)

---
### Student Tasks Addressed:
1. Capture vehicle images using surveillance cameras.
2. Identify and extract the number plate region (License plate localization and deskewing).
3. Apply OCR to recognise the characters and numbers (Character segmentation and pattern classification).
4. Classify and match the extracted number plate with registered vehicles (Syntactic disambiguation and database lookup).
5. Discuss challenges caused by poor lighting, different fonts, and tilted number plates (CLAHE, affine rectification, and robust thresholding).
6. Suggest how the system can be integrated with automatic parking payment (Ticketing, dynamic fee calculation, shopping mall discounts, FASTag auto-debit).
""")

    # Setup & Imports
    md("### 1. Environment Setup & Core Dependencies")
    code("""
import os
import sys
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Add repository root to path
if '.' not in sys.path:
    sys.path.insert(0, '.')

from src.dataset_generator import generate_character_dataset, generate_vehicle_scene, generate_benchmark_dataset, CLASSES
from src.preprocessing import preprocess_pipeline, apply_clahe, bilateral_noise_filter, detect_vertical_edges
from src.plate_detector import PlateDetector
from src.character_segmenter import CharacterSegmenter
from src.ocr_engine import OCREngine, HOGFeatureExtractor
from src.postprocessor import disambiguate_license_plate, match_registered_database
from src.parking_system import SmartParkingSystem
from src.pipeline import ANPRPipeline

print("All Pattern Recognition modules successfully imported!")
""")

    # Problem Formulation
    md("""
### 2. Problem Understanding & Mathematical Formulation

At busy shopping malls and commercial facilities, manual recording of vehicle registration numbers creates massive queue bottlenecks, human transcription errors, and revenue leakage. 

From a **Pattern Recognition** perspective, this problem is modeled as a multi-stage hierarchical decision process:
1. **Image Space $\\mathcal{I}$**: Surveillance frame $I(x, y) \\in \\mathbb{R}^{H \\times W \\times 3}$.
2. **Candidate Extraction**: Morphological operator $M(I) = I \\bullet B - I \\circ B$ and Sobel vertical gradient $G_x = I * S_x$.
3. **Bounding Box Decision**: Geometric decision function $f_{geom}(\\text{contour}) = w/h \\in [2.5, 5.5] \\wedge A_{rect} > \\theta_A$.
4. **Deskewing Homography**: Affine transformation matrix $T_{\\theta}$ rectifying rotated plates.
5. **Character Feature Vector**: Histogram of Oriented Gradients (HOG) mapping $x \\in \\mathbb{R}^{28 \\times 28} \\to \\phi(x) \\in \\mathbb{R}^{324}$.
6. **Bayesian Decision Rule**:
   $$\\hat{\\omega} = \\arg\\max_{\\omega_i} P(\\omega_i | x) = \\arg\\max_{\\omega_i} p(x | \\omega_i) P(\\omega_i)$$
   where loss $\\lambda_{ij}$ is minimized over 36 classes (0-9, A-Z).
""")

    # Dataset Generation & Visualization
    md("### 3. Dataset Collection & Environmental Synthesis")
    code("""
# Generate benchmark vehicle dataset across varied conditions
manifest = generate_benchmark_dataset(output_dir="data/raw", num_samples=25)
df_manifest = pd.DataFrame(manifest)
print(f"Total benchmark surveillance frames generated: {len(df_manifest)}")
df_manifest[['filename', 'condition', 'plate_text', 'tilt_angle']].head(10)
""")

    code("""
# Visualize samples under diverse environmental conditions
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
sample_conditions = ['normal', 'low_light', 'tilted', 'noisy']

for ax, cond in zip(axes, sample_conditions):
    row = next(item for item in manifest if item['condition'] == cond)
    img_bgr = cv2.imread(row['image_path'])
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Condition: {cond.upper()}\\nPlate: {row['plate_text']}", fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()
""")

    # Preprocessing & Localization
    md("""
### 4. Preprocessing & Plate Localization (Sobel + CLAHE + Morphology)

We apply:
1. **CLAHE**: Adaptively amplifies local contrast in dark/night surveillance frames.
2. **Bilateral Filtering**: Preserves character edges while removing high-ISO sensor noise.
3. **Vertical Sobel Operator**: Filters for high-frequency vertical strokes characteristic of license plates.
4. **Morphological Closing**: Merges character edges into a unified rectangular candidate blob.
""")
    code("""
sample_path = manifest[0]['image_path']
img = cv2.imread(sample_path)
prep = preprocess_pipeline(img)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Original Frame", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(prep['clahe'], cmap='gray')
axes[1].set_title("2. CLAHE Contrast", fontweight='bold')
axes[1].axis('off')

axes[2].imshow(prep['sobel'], cmap='gray')
axes[2].set_title("3. Vertical Sobel", fontweight='bold')
axes[2].axis('off')

axes[3].imshow(prep['candidate_mask'], cmap='gray')
axes[3].set_title("4. Morphological Mask", fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
plt.show()
""")

    # Deskewing & Character Segmentation
    md("""
### 5. Plate Deskewing & Adaptive Character Segmentation

Tilted plates caused by camera angle or vehicle trajectory are mathematically rectified using:
$$M = \\begin{bmatrix} \\alpha & \\beta & (1-\\alpha)c_x - \\beta c_y \\\\ -\\beta & \\alpha & \\beta c_x + (1-\\alpha)c_y \\end{bmatrix}$$
where $\\alpha = \\cos\\theta, \\beta = \\sin\\theta$.
""")
    code("""
detector = PlateDetector()
det_result = detector.detect(img, prep['candidate_mask'])
plate_crop = det_result['plate_crop']

segmenter = CharacterSegmenter()
chars, seg_dbg = segmenter.segment(plate_crop)

fig, axes = plt.subplots(1, 2, figsize=(12, 3))
axes[0].imshow(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Extracted & Deskewed Plate (Angle: {det_result['angle']:.1f}°)", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(seg_dbg['binary'], cmap='gray')
axes[1].set_title(f"Adaptive Gaussian Binarization ({len(chars)} Chars)", fontweight='bold')
axes[1].axis('off')
plt.show()

# Display segmented character glyphs (28x28 normalized)
fig, axes = plt.subplots(1, len(chars), figsize=(14, 2))
for idx, ch in enumerate(chars):
    axes[idx].imshow(ch['char_img'], cmap='gray')
    axes[idx].set_title(f"#{idx+1}")
    axes[idx].axis('off')
plt.tight_layout()
plt.show()
""")

    # OCR Model Training & Benchmarking
    md("""
### 6. Character Classification: Algorithm Comparison

We train and evaluate four distinct Pattern Recognition models:
1. **Template Matching**: Normalized Cross-Correlation prototype baseline.
2. **k-Nearest Neighbors (k-NN)**: Non-parametric instance-based classifier.
3. **Multi-Layer Perceptron (MLP)**: Feedforward neural network.
4. **Support Vector Machine (SVM)**: Maximum-margin hyperplane with RBF kernel and HOG feature descriptors.
""")
    code("""
ocr = OCREngine(models_dir="data/models")
metrics = ocr.train_models(samples_per_class=35)

df_benchmark = pd.read_csv("outputs/ocr_benchmark_metrics.csv")
print("--- OCR MODEL BENCHMARK ---")
display(df_benchmark)
""")

    code("""
# Visualize Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_benchmark))
w = 0.2
ax.bar(x - w*1.5, df_benchmark['Accuracy (%)'], w, label='Accuracy', color='#1E40AF')
ax.bar(x - w*0.5, df_benchmark['Precision (%)'], w, label='Precision', color='#0284C7')
ax.bar(x + w*0.5, df_benchmark['Recall (%)'], w, label='Recall', color='#0D9488')
ax.bar(x + w*1.5, df_benchmark['F1-Score (%)'], w, label='F1-Score', color='#F97316')
ax.set_xticks(x)
ax.set_xticklabels(df_benchmark['Model'], fontweight='bold')
ax.set_ylabel("Percentage (%)", fontweight='bold')
ax.set_title("Comparative Analysis of Pattern Recognition Classifiers", fontweight='bold')
ax.set_ylim(50, 105)
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
""")

    # Confusion Matrix
    md("### 7. Confusion Matrix & Bayes Error Analysis")
    code("""
if os.path.exists("outputs/figures/confusion_matrix.png"):
    cm_img = Image.open("outputs/figures/confusion_matrix.png")
    plt.figure(figsize=(12, 10))
    plt.imshow(cm_img)
    plt.axis('off')
    plt.title("HOG-SVM Alphanumeric Confusion Matrix (36 Classes)", fontweight='bold')
    plt.show()
""")

    # Smart Parking System Integration
    md("""
### 8. Real-World Smart Parking Solution & Automated Billing Integration

We simulate the complete commercial shopping mall workflow:
- **Vehicle Entry Gate**: Camera reads plate, checks VIP / Monthly pass whitelist, assigns slot, and triggers boom barrier open.
- **Vehicle Exit Gate**: Computes exact duration, calculates tiered parking fee, applies shopping mall retail voucher discounts, and generates electronic FASTag receipts.
""")
    code("""
pipeline = ANPRPipeline(model_type="svm")

# Step 1: Process Entry Gate
entry_res = pipeline.process(sample_path, action="entry")
ticket = entry_res['parking_result']

print("==================================================")
print("             SMART PARKING ENTRY TICKET           ")
print("==================================================")
print(f"Vehicle Number : {ticket['plate_number']}")
print(f"Permittee      : {ticket['owner_name']} ({ticket['category']})")
print(f"Assigned Slot  : {ticket['slot_id']}")
print(f"Timestamp      : {ticket['entry_time']}")
print(f"Barrier Status : [{ticket['barrier']}]")
print(f"System Message : {ticket['message']}")
print("==================================================")
""")

    code("""
# Step 2: Process Exit Gate & Automated Payment (Simulated 145 mins stay with Rs. 650 Mall Bill)
exit_res = pipeline.process(
    sample_path, 
    action="exit", 
    mall_spent=650.0, 
    simulated_duration_mins=145.0
)
receipt = exit_res['parking_result']

print("==================================================")
print("          METROPOLIS MALL PARKING RECEIPT         ")
print("==================================================")
print(f"Receipt ID       : REC-{receipt['log_id']:05d}")
print(f"Vehicle Number   : {receipt['plate_number']}")
print(f"Permit Category  : {receipt['category']}")
print(f"Parked Slot      : {receipt['slot_id']}")
print(f"Duration Parked  : {receipt['duration_formatted']} ({receipt['duration_minutes']} mins)")
print("--------------------------------------------------")
print(f"Base Parking Fee : Rs. {receipt['base_fee']:.2f}")
print(f"Mall Discount    : -Rs. {receipt['discount_amount']:.2f}")
print(f"TOTAL AMOUNT PAID: Rs. {receipt['final_amount']:.2f}")
print(f"Payment Method   : {receipt['payment_method']} (Electronic Auto-Debit)")
print(f"Payment Status   : {receipt['payment_status']}")
print(f"Exit Barrier     : [{receipt['barrier']}]")
print("==================================================")
""")

    # Environmental Robustness
    md("### 9. Environmental Robustness Across Real-World Surveillance Challenges")
    code("""
df_env = pd.read_csv("outputs/environmental_benchmark.csv")
print("--- PERFORMANCE ACROSS REAL-WORLD OPERATING CONDITIONS ---")
display(df_env)

if os.path.exists("outputs/figures/environmental_robustness.png"):
    env_img = Image.open("outputs/figures/environmental_robustness.png")
    plt.figure(figsize=(10, 5))
    plt.imshow(env_img)
    plt.axis('off')
    plt.show()
""")

    # Save to file
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.14.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = "ANPR_Smart_Parking_System.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"Notebook successfully written to {out_path} ({len(cells)} cells).")

if __name__ == "__main__":
    build_notebook()
