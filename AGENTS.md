# 🤖 AGENTS.md — Autonomous Agent Architecture & Developer Guide

## System Name: Automatic Number Plate Recognition (ANPR) Smart Parking System
**Course:** Pattern Recognition  
**Domain:** Computer Vision, Statistical Pattern Classification, Intelligent Transportation Systems

---

## 1. Architectural Philosophy: Multi-Agent Pipeline Decomposition

The ANPR Smart Parking System is designed as a collaborative, multi-agent pipeline where each specialized agent encapsulates a distinct level of abstraction in the pattern recognition hierarchy. Each agent operates with defined inputs, invariant constraints, transformation kernels, and failure-recovery protocols.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AGENT COLLABORATION GRAPH                       │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │  1. Sensory Acquisition Agent │
                   └───────────────┬───────────────┘
                                   │ Raw Frame I(x, y)
                                   ▼
                   ┌───────────────────────────────┐
                   │  2. Vision Preprocessing Agent│
                   └───────────────┬───────────────┘
                                   │ Enhanced Mask & Gradient
                                   ▼
                   ┌───────────────────────────────┐
                   │ 3. Plate Localization Agent   │
                   └───────────────┬───────────────┘
                                   │ Rectified Plate Crop
                                   ▼
                   ┌───────────────────────────────┐
                   │ 4. Character Segmenter Agent  │
                   └───────────────┬───────────────┘
                                   │ Standardized 28x28 Glyphs
                                   ▼
                   ┌───────────────────────────────┐
                   │ 5. HOG-SVM Pattern OCR Agent  │
                   └───────────────┬───────────────┘
                                   │ Raw Alphanumeric Strings & Conf
                                   ▼
                   ┌───────────────────────────────┐
                   │ 6. Syntactic Disambiguator    │
                   └───────────────┬───────────────┘
                                   │ Verified Plate Number
                                   ▼
                   ┌───────────────────────────────┐
                   │ 7. Smart Parking & Billing    │
                   │    Automation Agent           │
                   └───────────────────────────────┘
```

---

## 2. Agent Specifications & Responsibilities

### Agent 1: Sensory Acquisition Agent
- **Source Module:** `src/dataset_generator.py` / Camera Stream Ingestion
- **Role:** Interfaces with physical or synthetic surveillance video streams.
- **Input:** Camera sensor stream ($640 \times 480$, 30 FPS, BGR color space).
- **Outputs:** Timestamped surveillance frames $I(x, y) \in \mathbb{R}^{H \times W \times 3}$.
- **Invariant:** Maintains temporal consistency and tags environmental metadata (Normal Daylight, Low-Light/Night, Tilted Skew, Noise, Glare).

### Agent 2: Vision Preprocessing & Contrast Enhancement Agent
- **Source Module:** `src/preprocessing.py`
- **Role:** Prepares raw sensory frames for localization by eliminating sensor noise while maximizing edge contrast.
- **Sub-Pipelines:**
  - `convert_to_grayscale()`: Dimensionality reduction from 3-channel RGB to single-channel intensity.
  - `apply_clahe(clip_limit=3.0, tile_grid=(8, 8))`: Prevents noise amplification in dark or shadowed environments while amplifying local character stroke contrast.
  - `bilateral_noise_filter(d=9, sigma_color=75, sigma_space=75)`: Smooths high-ISO sensor grain while preserving sharp character boundaries.
  - `detect_vertical_edges()`: Vertical Sobel filtering ($G_x$) isolates high-frequency character strokes while suppressing horizontal bumper contours.
  - `morphological_close_to_blob()`: Rectangular structuring element aggregates discrete strokes into a unified candidate plate blob.

### Agent 3: Plate Localization & Deskewing Agent
- **Source Module:** `src/plate_detector.py`
- **Role:** Locates candidate license plates and geometrically rectifies perspective/tilt distortion.
- **Decision Heuristics:**
  - Evaluates contours using Aspect Ratio $AR \in [2.2, 6.0]$, Area $A \in [2500, 45000]$, and Extent $E > 0.40$.
  - Computes Minimum Area Bounding Box (`cv2.minAreaRect`).
  - Calculates skew angle $\theta$ and constructs an affine rotation matrix $M_\theta$ to warp the plate into a horizontal rectilinear crop.

### Agent 4: Character Segmentation Agent
- **Source Module:** `src/character_segmenter.py`
- **Role:** Isolates individual alphanumeric character glyphs from the deskewed plate crop.
- **Sub-Pipelines:**
  - `binarize_plate()`: Applies Adaptive Gaussian Thresholding with polarity verification (ensuring foreground strokes are white $\text{value}=255$ and background is black $\text{value}=0$).
  - `IND Strip Suppression`: Discards the blue European/Indian regional strip ($x < 0.13 \cdot W_{plate}$).
  - Aspect Ratio Preservation: Extracted characters are zero-padded to square aspect ratio and resized to standardized $28 \times 28$ normalized tensors.

### Agent 5: Feature Extraction & Pattern OCR Agent
- **Source Module:** `src/ocr_engine.py`
- **Role:** Extracts invariant statistical descriptors and performs statistical pattern classification across 36 alphanumeric classes (0–9, A–Z).
- **Core Models:**
  - `HOGFeatureExtractor`: Partitions glyphs into $7 \times 7$ spatial cells with 9 orientation bins and $2 \times 2$ block $L_2\text{-Hys}$ normalization (324-dimensional feature vector).
  - **Support Vector Machine (SVM):** Maximum-margin separation with Radial Basis Function (RBF) kernel and calibrated Platt scaling.
  - **Multi-Layer Perceptron (MLP):** Feedforward connectionist architecture `Dense(128, ReLU) -> Dense(64, ReLU) -> Softmax(36)`.
  - **k-Nearest Neighbor (k-NN):** Non-parametric instance-based classifier ($k=3$).
  - **Template Matching:** Baseline normalized cross-correlation against mean prototype glyphs.

### Agent 6: Syntactic Post-Processing & Disambiguation Agent
- **Source Module:** `src/postprocessor.py`
- **Role:** Minimizes Bayes classification risk by enforcing grammatical priors and fuzzy database matching.
- **Ambiguity Resolution:**
  - Positional Syntactic Domain Priors:
    - Indices 0–1 (State Code): Strictly mapped to Letters ('0' $\to$ 'O', '8' $\to$ 'B', '1' $\to$ 'I').
    - Indices 2–3 (RTO District Code): Strictly mapped to Digits ('O' $\to$ '0', 'B' $\to$ '8').
    - Last 4 Indices: Strictly mapped to Digits.
  - Levenshtein Edit-Distance: Performs fuzzy matching against registered permittees to correct isolated single-character substitution errors.

### Agent 7: Smart Parking Automation & Billing Agent
- **Source Module:** `src/parking_system.py`
- **Role:** Connects perception with physical barrier actuators and electronic commercial billing.
- **Capabilities:**
  - Manages SQLite relational database tables (`parking_slots`, `registered_vehicles`, `parking_logs`).
  - Dynamic slot allocation (VIP priority, EV charging, general parking).
  - Dynamic tiered tariff calculation with shopping mall retail voucher discounts.
  - Generates itemized electronic FASTag and UPI digital receipts with barrier state control.

---

## 3. Developer & Autonomous Agent Execution Guide

### Environment Setup
```powershell
pip install -r requirements.txt
```

### Running System Execution Workflows
1. **End-to-End Single Vehicle Gate Simulation (CLI):**
   ```powershell
   python main.py
   ```
2. **Batch Evaluation on Multi-Condition Surveillance Dataset:**
   ```powershell
   python main.py --batch
   ```
3. **Launch Interactive Streamlit Web Dashboard:**
   ```powershell
   streamlit run web_app.py
   ```
4. **Re-train Models & Generate Publication Benchmark Plots:**
   ```powershell
   python evaluation/benchmark.py
   ```
5. **Interactive Step-by-Step Jupyter Notebook:**
   Launch VS Code or JupyterLab and open `ANPR_Smart_Parking_System.ipynb`.

---

## 4. Codebase Conventions & Guidelines for AI Agents

When modifying, extending, or maintaining this codebase:
- **Never hardcode paths:** Always resolve paths relative to `os.path.dirname(__file__)` or project root.
- **Preserve Aspect Ratios:** Never resize extracted characters directly to $28 \times 28$ without padding to square first; direct anisotropic scaling distorts thin glyphs like '1' and 'I'.
- **Bilateral Filter vs Gaussian:** Never replace bilateral filtering with plain Gaussian blur in preprocessing; Gaussian blur degrades character stroke edges required for segmentation.
- **Thread Safety in Streamlit:** Use `@st.cache_resource` for model loading to avoid redundant disk I/O and weight reloads.
- **Database Hygiene:** Always commit transactions and close SQLite connections cleanly in `finally` blocks or context managers.
