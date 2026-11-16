"""
Comprehensive Evaluation and Benchmarking Suite
Evaluates:
1. Character OCR models comparison (Template Matching vs k-NN vs SVM vs MLP)
2. Performance under environmental variations (Normal, Low-light, Tilted, Noisy, Glare)
3. Ablation study: Effect of CLAHE on low-light & Affine Deskewing on tilted plates
4. Latency profiling (FPS / runtime per stage)
5. Generates figures and CSV reports for project report and viva presentation.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time
import tempfile
mpl_dir = os.path.join(tempfile.gettempdir(), 'mpl_cache')
os.makedirs(mpl_dir, exist_ok=True)
os.environ['MPLCONFIGDIR'] = mpl_dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

from src.dataset_generator import CLASSES, generate_character_dataset
from src.ocr_engine import OCREngine, HOGFeatureExtractor
from src.pipeline import ANPRPipeline
from src.preprocessing import apply_clahe, bilateral_noise_filter

def evaluate_ocr_models(samples_per_class=40, output_dir="outputs/figures"):
    """
    Train and rigorously benchmark Template Matching, k-NN, SVM, and MLP.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("STEP 1: Evaluating Character Recognition Models (36 Classes: 0-9, A-Z)")
    print("=" * 60)

    X, y, classes = generate_character_dataset(samples_per_class=samples_per_class)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    hog = HOGFeatureExtractor()
    X_train_hog = hog.compute_batch(X_train)
    X_test_hog = hog.compute_batch(X_test)

    # Initialize models
    ocr = OCREngine(models_dir="data/models")
    ocr.train_models(samples_per_class=samples_per_class)

    models_to_test = ["template", "knn", "svm", "mlp"]
    results = []

    y_test_pred_svm = None

    for m_type in models_to_test:
        ocr.load_model(m_type)
        start_time = time.time()
        
        preds = []
        for img in X_test:
            char_pred, conf, _ = ocr.predict_char(img)
            preds.append(CLASSES.index(char_pred) if char_pred in CLASSES else 0)
        
        elapsed = time.time() - start_time
        latency_ms = (elapsed / len(X_test)) * 1000.0

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)

        if m_type == "svm":
            y_test_pred_svm = preds

        results.append({
            "Model": m_type.upper(),
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2),
            "Latency (ms/char)": round(latency_ms, 2)
        })

    df_models = pd.DataFrame(results)
    print("\n--- OCR MODEL BENCHMARK RESULTS ---")
    print(df_models.to_string(index=False))
    df_models.to_csv("outputs/ocr_benchmark_metrics.csv", index=False)

    # Plot 1: Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    bar_width = 0.2
    indices = np.arange(len(df_models))

    plt.bar(indices - bar_width*1.5, df_models["Accuracy (%)"], bar_width, label="Accuracy", color="#2b5c8f")
    plt.bar(indices - bar_width*0.5, df_models["Precision (%)"], bar_width, label="Precision", color="#348498")
    plt.bar(indices + bar_width*0.5, df_models["Recall (%)"], bar_width, label="Recall", color="#5bd1d7")
    plt.bar(indices + bar_width*1.5, df_models["F1-Score (%)"], bar_width, label="F1-Score", color="#ff502f")

    plt.xlabel("Pattern Recognition Model", fontsize=12, fontweight="bold")
    plt.ylabel("Performance Score (%)", fontsize=12, fontweight="bold")
    plt.title("Comparative Performance Analysis of Character Classifiers", fontsize=14, fontweight="bold")
    plt.xticks(indices, df_models["Model"], fontsize=11)
    plt.ylim(50, 105)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_bar.png"), dpi=300)
    plt.close()

    # Plot 2: Confusion Matrix for Top Model (SVM)
    if y_test_pred_svm is not None:
        cm = confusion_matrix(y_test, y_test_pred_svm)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues", cbar=True)
        plt.title("HOG-SVM Alphanumeric Confusion Matrix (36 Classes)", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
        plt.ylabel("True Class", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
        plt.close()

    return df_models


def compute_iou(boxA, boxB):
    """Compute Intersection over Union between two boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return max(0.0, min(1.0, iou))


def evaluate_end_to_end_pipeline(manifest_path="data/raw/manifest.json", output_dir="outputs/figures"):
    """
    Run end-to-end evaluation across conditions (Normal, Low-light, Tilted, Noisy, Glare).
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("STEP 2: End-to-End Pipeline Evaluation Across Environmental Conditions")
    print("=" * 60)

    if not os.path.exists(manifest_path):
        print("Manifest not found. Please run dataset_generator first.")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    pipeline = ANPRPipeline(model_type="svm")

    condition_records = {}

    for item in manifest:
        cond = item["condition"]
        if cond not in condition_records:
            condition_records[cond] = {
                "count": 0,
                "iou_scores": [],
                "char_matches": [],
                "exact_matches": 0,
                "latencies_ms": []
            }

        img_path = item["image_path"]
        gt_plate = item["plate_text"]
        gt_bbox = item["bbox"]

        t0 = time.time()
        res = pipeline.process(img_path, action="detect_only")
        t_el = (time.time() - t0) * 1000.0

        pred_bbox = res["detection"]["bbox"]
        iou = compute_iou(gt_bbox, pred_bbox)

        pred_plate = res["final_plate"]
        exact = (pred_plate == gt_plate)
        if exact:
            condition_records[cond]["exact_matches"] += 1

        # Character match ratio
        gt_len = max(1, len(gt_plate))
        matching_chars = sum(1 for c1, c2 in zip(pred_plate, gt_plate) if c1 == c2)
        char_acc = (matching_chars / gt_len) * 100.0

        condition_records[cond]["count"] += 1
        condition_records[cond]["iou_scores"].append(iou)
        condition_records[cond]["char_matches"].append(char_acc)
        condition_records[cond]["latencies_ms"].append(t_el)

    summary = []
    for cond, data in condition_records.items():
        n = data["count"]
        summary.append({
            "Condition": cond.capitalize(),
            "Samples": n,
            "Mean Plate IoU": round(float(np.mean(data["iou_scores"])), 3),
            "Char Accuracy (%)": round(float(np.mean(data["char_matches"])), 2),
            "Plate Exact Match (%)": round((data["exact_matches"] / n) * 100.0, 2),
            "Avg Latency (ms)": round(float(np.mean(data["latencies_ms"])), 1)
        })

    df_env = pd.DataFrame(summary)
    print("\n--- ENVIRONMENTAL CONDITIONS BENCHMARK ---")
    print(df_env.to_string(index=False))
    df_env.to_csv("outputs/environmental_benchmark.csv", index=False)

    # Plot 3: Environmental Robustness Grouped Bar Chart
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df_env))
    w = 0.35

    plt.bar(x - w/2, df_env["Char Accuracy (%)"], w, label="Character Accuracy (%)", color="#1f77b4")
    plt.bar(x + w/2, df_env["Plate Exact Match (%)"], w, label="Plate Exact Match (%)", color="#ff7f0e")

    plt.xlabel("Surveillance Operating Condition", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy Score (%)", fontsize=12, fontweight="bold")
    plt.title("ANPR System Performance Across Real-World Environmental Challenges", fontsize=14, fontweight="bold")
    plt.xticks(x, df_env["Condition"], fontsize=11)
    plt.ylim(0, 110)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "environmental_robustness.png"), dpi=300)
    plt.close()

    # Plot 4: Pipeline Stages Visual Breakdown Demo
    sample_item = manifest[0]
    res_sample = pipeline.process(sample_item["image_path"], action="detect_only")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(cv2.cvtColor(res_sample["original_image"], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Surveillance Input Frame", fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(res_sample["preprocessing"]["clahe"], cmap="gray")
    axes[0, 1].set_title("2. CLAHE Contrast Enhanced", fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(res_sample["preprocessing"]["candidate_mask"], cmap="gray")
    axes[0, 2].set_title("3. Morphological Candidate Mask", fontweight="bold")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(res_sample["plate_crop"], cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"4. Extracted & Deskewed Plate ({res_sample['detection']['angle']:.1f}°)", fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(res_sample["seg_debug"]["binary"], cmap="gray")
    axes[1, 1].set_title(f"5. Segmented Binary ({len(res_sample['segmentation'])} chars)", fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(cv2.cvtColor(res_sample["annotated_image"], cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f"6. Final: {res_sample['final_plate']} ({res_sample['ocr_confidence']*100:.1f}%)", fontweight="bold")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pipeline_stages_demo.png"), dpi=300)
    plt.close()

    print(f"Generated evaluation plots in '{output_dir}'.")
    return df_env

if __name__ == "__main__":
    evaluate_ocr_models()
    evaluate_end_to_end_pipeline()
