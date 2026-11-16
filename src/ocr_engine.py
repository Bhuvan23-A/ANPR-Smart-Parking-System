"""
Pattern Recognition Optical Character Recognition (OCR) Engine
Implements and compares multiple Pattern Recognition paradigms:
1. Baseline: Normalized Correlation Template Matching
2. Instance-based: k-Nearest Neighbor (k-NN)
3. Margin-based: Support Vector Machine (SVM with RBF/Linear Kernel & Platt Scaling)
4. Connectionist: Multi-Layer Perceptron (MLP Neural Network)
Extracts Histogram of Oriented Gradients (HOG) and Structural Statistical features.
"""

import os
import joblib
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.dataset_generator import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, generate_character_dataset

class HOGFeatureExtractor:
    """
    Extracts Histogram of Oriented Gradients (HOG) descriptors from 28x28 character images.
    """
    def __init__(self, win_size=(28, 28), block_size=(14, 14), block_stride=(7, 7), cell_size=(7, 7), nbins=9):
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins

    def compute(self, char_img):
        """
        Compute HOG descriptor vector for a single 28x28 grayscale image.
        char_img can be float32 in [0, 1] or uint8 in [0, 255].
        """
        if char_img.dtype != np.uint8:
            img_u8 = (char_img * 255).astype(np.uint8)
        else:
            img_u8 = char_img

        # Manual or Sobel gradient HOG
        # Compute horizontal and vertical gradients
        gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # Discretize angles into nbins (0 to 180 degrees)
        bin_width = 180.0 / self.nbins
        ang = np.mod(ang, 180.0)

        # Partition into cells (4x4 cells of 7x7 pixels)
        cells_x = self.win_size[0] // self.cell_size[0]
        cells_y = self.win_size[1] // self.cell_size[1]
        hist = np.zeros((cells_y, cells_x, self.nbins), dtype=np.float32)

        for cy in range(cells_y):
            for cx in range(cells_x):
                y_start = cy * self.cell_size[1]
                y_end = y_start + self.cell_size[1]
                x_start = cx * self.cell_size[0]
                x_end = x_start + self.cell_size[0]

                cell_mag = mag[y_start:y_end, x_start:x_end]
                cell_ang = ang[y_start:y_end, x_start:x_end]

                for b in range(self.nbins):
                    b_start = b * bin_width
                    b_end = (b + 1) * bin_width
                    mask = (cell_ang >= b_start) & (cell_ang < b_end)
                    hist[cy, cx, b] = np.sum(cell_mag[mask])

        # Block normalization (L2-Hys)
        features = []
        for by in range(cells_y - 1):
            for bx in range(cells_x - 1):
                block = hist[by:by+2, bx:bx+2, :].flatten()
                norm = np.linalg.norm(block) + 1e-6
                block_norm = np.clip(block / norm, 0, 0.2)
                block_norm = block_norm / (np.linalg.norm(block_norm) + 1e-6)
                features.extend(block_norm)

        return np.array(features, dtype=np.float32)

    def compute_batch(self, X):
        """Compute HOG for a batch of images shape (N, 28, 28)."""
        return np.array([self.compute(img) for img in X], dtype=np.float32)


class OCREngine:
    def __init__(self, model_type="svm", models_dir="data/models"):
        """
        Args:
            model_type (str): 'svm', 'knn', 'mlp', or 'template'
            models_dir (str): Directory where trained weights are saved
        """
        self.model_type = model_type
        self.models_dir = models_dir
        self.hog = HOGFeatureExtractor()
        self.model = None
        self.template_prototypes = {}
        os.makedirs(models_dir, exist_ok=True)

    def train_models(self, samples_per_class=45):
        """
        Train all Pattern Recognition classifiers on synthetic augmented characters.
        """
        print(f"Generating character dataset with {samples_per_class} samples/class...")
        X, y, classes = generate_character_dataset(samples_per_class=samples_per_class)
        
        # Split train / test (80/20 stratified)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Extracting HOG feature descriptors...")
        X_train_hog = self.hog.compute_batch(X_train)
        X_test_hog = self.hog.compute_batch(X_test)
        print(f"HOG feature dimension: {X_train_hog.shape[1]}")

        # 1. Train SVM (RBF Kernel with Platt Scaling for posterior probability)
        print("Training Support Vector Machine (SVM - RBF Kernel)...")
        svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
        svm.fit(X_train_hog, y_train)
        svm_acc = accuracy_score(y_test, svm.predict(X_test_hog))
        print(f"-> SVM Test Accuracy: {svm_acc * 100:.2f}%")
        joblib.dump(svm, os.path.join(self.models_dir, "svm_model.joblib"))

        # 2. Train k-NN
        print("Training k-Nearest Neighbors (k-NN, k=3)...")
        knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
        knn.fit(X_train_hog, y_train)
        knn_acc = accuracy_score(y_test, knn.predict(X_test_hog))
        print(f"-> k-NN Test Accuracy: {knn_acc * 100:.2f}%")
        joblib.dump(knn, os.path.join(self.models_dir, "knn_model.joblib"))

        # 3. Train MLP Neural Network
        print("Training Multi-Layer Perceptron (MLP)...")
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=250, activation="relu", random_state=42)
        mlp.fit(X_train_hog, y_train)
        mlp_acc = accuracy_score(y_test, mlp.predict(X_test_hog))
        print(f"-> MLP Test Accuracy: {mlp_acc * 100:.2f}%")
        joblib.dump(mlp, os.path.join(self.models_dir, "mlp_model.joblib"))

        # 4. Compute Mean Template Prototypes
        print("Computing Template Prototypes for baseline...")
        templates = {}
        for c_idx in range(len(CLASSES)):
            c_imgs = X_train[y_train == c_idx]
            mean_proto = np.mean(c_imgs, axis=0)
            templates[c_idx] = mean_proto
        joblib.dump(templates, os.path.join(self.models_dir, "templates.joblib"))

        # Load selected model
        self.load_model(self.model_type)

        metrics = {
            "svm_accuracy": svm_acc,
            "knn_accuracy": knn_acc,
            "mlp_accuracy": mlp_acc,
            "hog_features": X_train_hog.shape[1],
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        return metrics

    def load_model(self, model_type="svm"):
        """Load trained model weights from disk."""
        self.model_type = model_type
        model_path = os.path.join(self.models_dir, f"{model_type}_model.joblib")
        template_path = os.path.join(self.models_dir, "templates.joblib")

        if os.path.exists(template_path):
            self.template_prototypes = joblib.load(template_path)

        if model_type == "template":
            return True

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            return True
        else:
            print(f"Model {model_path} not found. Training models now...")
            self.train_models()
            return True

    def predict_char(self, char_img_28x28):
        """
        Classify a single 28x28 normalized character image.
        
        Returns:
            predicted_char (str): '0'-'9', 'A'-'Z'
            confidence (float): Probability in [0.0, 1.0]
            top3 (list): [(char, prob), ...]
        """
        if self.model is None and self.model_type != "template":
            self.load_model(self.model_type)

        # Baseline Template Matching
        if self.model_type == "template":
            best_score = -1.0
            best_idx = 0
            scores = []
            for c_idx, proto in self.template_prototypes.items():
                # Normalized Cross-Correlation
                res = cv2.matchTemplate(
                    char_img_28x28.astype(np.float32),
                    proto.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED
                )
                score = float(res[0][0])
                scores.append((c_idx, score))
                if score > best_score:
                    best_score = score
                    best_idx = c_idx
            
            scores.sort(key=lambda x: x[1], reverse=True)
            top3 = [(IDX_TO_CLASS[idx], max(0.0, s)) for idx, s in scores[:3]]
            return IDX_TO_CLASS[best_idx], max(0.0, best_score), top3

        # HOG Feature Extraction
        hog_feat = self.hog.compute(char_img_28x28).reshape(1, -1)

        # Classifier inference
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(hog_feat)[0]
            top_indices = np.argsort(probs)[::-1][:3]
            best_idx = top_indices[0]
            confidence = float(probs[best_idx])
            top3 = [(IDX_TO_CLASS[i], float(probs[i])) for i in top_indices]
        else:
            pred_idx = self.model.predict(hog_feat)[0]
            best_idx = int(pred_idx)
            confidence = 0.85
            top3 = [(IDX_TO_CLASS[best_idx], 0.85)]

        return IDX_TO_CLASS[best_idx], confidence, top3

    def recognize_plate(self, segmented_chars):
        """
        Recognize an ordered list of segmented character dicts.
        
        Returns:
            raw_text (str): Extracted license plate string
            mean_conf (float): Average recognition confidence
            char_details (list): Detailed predictions per character
        """
        if not segmented_chars:
            return "", 0.0, []

        recognized_chars = []
        confidences = []
        details = []

        for item in segmented_chars:
            char_img = item["char_img"]
            pred_char, conf, top3 = self.predict_char(char_img)
            recognized_chars.append(pred_char)
            confidences.append(conf)
            details.append({
                "char": pred_char,
                "confidence": conf,
                "bbox": item["bbox"],
                "top3": top3
            })

        raw_text = "".join(recognized_chars)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        return raw_text, mean_conf, details
