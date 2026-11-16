# 🚗 Automatic Number Plate Recognition (ANPR) for Smart Parking

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.14-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Pattern%20Recognition-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-red.svg)](https://streamlit.io/)
[![Course](https://img.shields.io/badge/Course-Pattern%20Recognition-purple.svg)](#)

An intelligent, end-to-end Pattern Recognition and Computer Vision system designed to automate vehicle entry/exit gates in commercial facilities (shopping malls, office complexes, airports). It localizes vehicle license plates, deskews rotated plates, segments alphanumeric glyphs, classifies characters using HOG-SVM, resolves OCR visual ambiguities via syntactic priors, and manages parking bay allocation and automated FASTag electronic billing.

---

## 📌 System Architecture

```
[Surveillance Camera Frame]
            │
            ▼
┌──────────────────────────────────────────────┐
│ Stage 1: Sensory Space & Preprocessing       │
│ - Grayscale Conversion                       │
│ - CLAHE Local Contrast Equalization          │
│ - Bilateral Edge-Preserving Filter           │
│ - Vertical Sobel Edge Gradient Filter        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 2: Plate Localization & Deskewing      │
│ - Morphological Closing (Blob Formation)     │
│ - Geometric Contour Filtering (AR, Extent)   │
│ - Minimum Area Bounding Box Affine Deskewing │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 3: Character Segmentation              │
│ - Adaptive Gaussian Binarization             │
│ - Blue IND Strip / Frame Border Removal      │
│ - Vertical Projection Profiling              │
│ - Aspect Ratio Preservation (Pad to 28x28)   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 4: Feature Extraction & Classification │
│ - Histogram of Oriented Gradients (HOG)      │
│ - Multi-Model Evaluation (SVM, MLP, k-NN)    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 5: Post-Processing & Bayes Priors      │
│ - Positional Syntactic Disambiguation        │
│   ('0' vs 'O', '8' vs 'B', '1' vs 'I')       │
│ - Levenshtein Distance Whitelist Matching    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 6: Smart Parking Management            │
│ - Entry Gate: Dynamic Slot Allocation & Gate │
│ - Exit Gate: Tiered Tariff, Mall Discounts   │
│ - Automated FASTag / UPI Digital Billing     │
└──────────────────────────────────────────────┘
```

---

## 📊 Experimental Results & Benchmarks

### 1. Character Classifier Comparison (36 Classes: 0–9, A–Z)
| Model Paradigm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Latency (ms/char) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Template Matching** | 55.00% | 78.35% | 55.00% | 59.94% | **1.62 ms** |
| **k-Nearest Neighbor (k-NN)** | 78.89% | 80.87% | 78.89% | 78.85% | 16.37 ms |
| **Multi-Layer Perceptron (MLP)** | 84.17% | 84.78% | 84.17% | 84.05% | 3.60 ms |
| **HOG-SVM (RBF Kernel)** | **87.50%** | **88.83%** | **87.50%** | **87.56%** | **4.33 ms** |

### 2. Environmental Robustness Across Conditions
- **Plate Localization IoU:** **0.890** (Daylight) and **0.894** (Night).
- **Night-time Accuracy:** Combining CLAHE with adaptive Gaussian binarization improved low-light character accuracy from 0% to **66.00%**.
- **Average Pipeline Latency:** **~130–200 ms per frame** (real-time throughput).

---

## 🚀 Quick-Start Guide

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/<your-username>/ANPR-Smart-Parking-System.git
cd ANPR-Smart-Parking-System
pip install -r requirements.txt
```

### 2. Interactive Web Dashboard
Launch the interactive Streamlit GUI:
```bash
streamlit run web_app.py
```
- Test live vehicle entry and exit gate simulations.
- Inspect intermediate Computer Vision stages (CLAHE, Sobel, deskewing, segmented glyphs).
- View live database records (`parking_slots`, `registered_vehicles`, `parking_logs`).

### 3. Command-Line Simulation
Run a single vehicle simulation (Entry ticket issuance & FASTag billing receipt):
```bash
python main.py
```
Run batch evaluation on benchmark surveillance scenes:
```bash
python main.py --batch
```

### 4. Jupyter Notebook
Open `ANPR_Smart_Parking_System.ipynb` to step through the code and inline visualizations.

### 5. Re-run Benchmarks
To train models from scratch, evaluate test splits, and export metrics and figures:
```bash
python evaluation/benchmark.py
```

---

## 📁 Repository Structure
```
├── src/
│   ├── dataset_generator.py      # Character & vehicle scene synthesizer
│   ├── preprocessing.py          # CLAHE, bilateral filter, Sobel edge operator
│   ├── plate_detector.py         # Contour aspect ratio filtering & affine deskewing
│   ├── character_segmenter.py    # Adaptive Gaussian binarization, CCA, padding
│   ├── ocr_engine.py             # HOG extractor, SVM, MLP, k-NN, Template Matching
│   ├── postprocessor.py          # Positional syntactic disambiguation & fuzzy match
│   ├── parking_system.py         # SQLite database, slot allocator, dynamic tariff
│   └── pipeline.py               # Master end-to-end pipeline coordinator
├── evaluation/
│   └── benchmark.py              # Comparative benchmark across models & conditions
├── data/
│   ├── raw/                      # Benchmark surveillance images + manifest.json
│   ├── models/                   # Serialized model checkpoints
│   └── parking.db                # SQLite database
├── outputs/
│   ├── figures/                  # Publication-quality charts (confusion matrix, ROC, bars)
│   └── *.csv                     # Benchmarking CSV metrics
├── web_app.py                    # Streamlit web dashboard
├── main.py                       # CLI driver
├── ANPR_Smart_Parking_System.ipynb # Interactive notebook
├── PROJECT_REPORT.md             # Detailed academic report (100-mark rubric)
└── VIVA_QUESTIONS_AND_ANSWERS.md # 14 in-depth viva questions & answers
```

---

## 👥 Contributors
Developed as part of the **Pattern Recognition** course project (Case Study 13).
