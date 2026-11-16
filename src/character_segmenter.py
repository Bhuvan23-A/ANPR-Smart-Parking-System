"""
Character Segmentation Module for Automatic Number Plate Recognition
Performs:
1. Binarization & Adaptive Thresholding
2. Blue IND strip & border artifact removal
3. Connected Component Analysis (CCA) and Contour filtering
4. Vertical Projection Profile verification
5. Left-to-right sorting and standardized (28x28) normalization
"""

import cv2
import numpy as np

class CharacterSegmenter:
    def __init__(self, target_size=(28, 28)):
        self.target_size = target_size

    def binarize_plate(self, plate_bgr):
        """
        Convert plate to clean binary image with white characters on black background.
        Uses CLAHE and Adaptive Gaussian Thresholding to handle uneven lighting,
        low-light, shadows, and reflective glare.
        """
        if len(plate_bgr.shape) == 3:
            gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_bgr.copy()

        # Enhance local character contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        norm = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

        # Adaptive Gaussian Thresholding (robust against local illumination variations)
        binary = cv2.adaptiveThreshold(
            norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9
        )

        # Polarity verification: In a valid plate, foreground characters occupy 15% - 45% of pixels
        white_ratio = np.mean(binary == 255)
        if white_ratio > 0.50:
            binary = cv2.bitwise_not(binary)

        # Remove outer frame borders (top, bottom, left, right edges)
        h, w = binary.shape
        binary[0:int(h * 0.08), :] = 0
        binary[int(h * 0.92):, :] = 0
        binary[:, 0:int(w * 0.03)] = 0
        binary[:, int(w * 0.97):] = 0

        # Small morphological noise cleaning
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel)

        return binary, gray

    def compute_vertical_projection(self, binary_img):
        """
        Compute Vertical Projection Profile: P(x) = sum_y I(x, y)
        Useful for analyzing character spacing and splitting touched characters.
        """
        v_proj = np.sum(binary_img, axis=0) / 255.0
        return v_proj

    def segment(self, plate_bgr):
        """
        Isolate each character from the license plate.
        
        Returns:
            segmented_chars (list of dict):
                - 'char_img': 28x28 normalized float32 [0, 1]
                - 'bbox': [x, y, w, h] in plate coordinates
                - 'aspect_ratio': w / h
            debug_info (dict):
                - 'binary': binarized plate image
                - 'v_proj': vertical projection profile
        """
        h_plate, w_plate = plate_bgr.shape[:2]
        if h_plate < 15 or w_plate < 50:
            return [], {}

        binary, gray = self.binarize_plate(plate_bgr)
        v_proj = self.compute_vertical_projection(binary)

        # Find candidate character contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_chars = []
        # Exclude left IND strip region (first ~13% of width)
        min_x_limit = int(w_plate * 0.13)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = float(w) / float(h)

            # Character geometric constraints:
            # 1. Height must be between 30% and 85% of plate height
            # 2. Width must be between 2% and 26% of plate width
            # 3. Aspect ratio between 0.10 and 1.25
            # 4. Must start after the blue IND strip
            if (
                0.28 * h_plate <= h <= 0.88 * h_plate and
                0.02 * w_plate <= w <= 0.26 * w_plate and
                0.10 <= aspect_ratio <= 1.25 and
                x >= min_x_limit and
                area > 20
            ):
                candidate_chars.append((x, y, w, h))

        # Sort left-to-right by x coordinate
        candidate_chars.sort(key=lambda b: b[0])

        # Filter overlapping or duplicate bounding boxes (Non-Maximum Suppression)
        filtered_boxes = []
        for box in candidate_chars:
            if not filtered_boxes:
                filtered_boxes.append(box)
                continue
            prev = filtered_boxes[-1]
            # Check overlap
            px, py, pw, ph = prev
            x, y, w, h = box
            overlap_x = max(0, min(px + pw, x + w) - max(px, x))
            if overlap_x > 0.6 * min(pw, w):
                # Pick the larger one
                if (w * h) > (pw * ph):
                    filtered_boxes[-1] = box
            else:
                filtered_boxes.append(box)

        # Prepare normalized 28x28 character images
        segmented_output = []
        for x, y, w, h in filtered_boxes:
            char_crop = binary[y:y+h, x:x+w]
            
            # Pad to square to prevent distortion during resize
            max_dim = max(w, h)
            square = np.zeros((max_dim, max_dim), dtype=np.uint8)
            pad_x = (max_dim - w) // 2
            pad_y = (max_dim - h) // 2
            square[pad_y:pad_y+h, pad_x:pad_x+w] = char_crop

            # Resize to target size (28x28)
            resized = cv2.resize(square, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize to [0.0, 1.0]
            norm_char = resized.astype(np.float32) / 255.0

            segmented_output.append({
                "char_img": norm_char,
                "bbox": (x, y, w, h),
                "aspect_ratio": float(w) / float(h)
            })

        debug_info = {
            "binary": binary,
            "gray": gray,
            "v_proj": v_proj,
            "num_candidates": len(segmented_output)
        }

        return segmented_output, debug_info
