"""
License Plate Detection, Localization, and Deskewing Module
Identifies the candidate number plate region using contour geometry,
aspect ratio filtering, solidity scoring, and rectifies tilted plates
using Minimum Area Bounding Box affine deskewing.
"""

import cv2
import numpy as np

class PlateDetector:
    def __init__(
        self,
        min_aspect_ratio=2.2,
        max_aspect_ratio=6.0,
        min_area=2500,
        max_area=45000,
        min_extent=0.40
    ):
        self.min_ar = min_aspect_ratio
        self.max_ar = max_aspect_ratio
        self.min_area = min_area
        self.max_area = max_area
        self.min_extent = min_extent

    def deskew_plate(self, image, min_area_rect):
        """
        Rectify / Deskew tilted license plate using affine rotation
        based on cv2.minAreaRect.
        
        Args:
            image (np.ndarray): Original vehicle image
            min_area_rect: ((cx, cy), (w, h), angle)
            
        Returns:
            rectified_plate (np.ndarray): Deskewed horizontal plate image
            angle (float): Corrected tilt angle
        """
        (center_x, center_y), (width, height), angle = min_area_rect

        # OpenCV minAreaRect returns angle in [-90, 0) or [0, 90) depending on version
        # Adjust dimensions if width < height
        if width < height:
            width, height = height, width
            angle += 90.0

        # Adjust angle range to [-45, 45]
        if angle > 45:
            angle -= 90.0
        elif angle < -45:
            angle += 90.0

        # Create rotation matrix around center
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        
        # Warp the full image
        h_img, w_img = image.shape[:2]
        rotated = cv2.warpAffine(image, M, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Crop the rectified bounding box
        x = int(center_x - width / 2)
        y = int(center_y - height / 2)
        w = int(width)
        h = int(height)
        
        # Guard bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        rectified_plate = rotated[y:y+h, x:x+w]
        return rectified_plate, angle

    def score_contour(self, contour, frame_shape):
        """
        Score a contour based on aspect ratio, area, rectangularity (extent),
        and vertical position in the vehicle frame.
        """
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if area < self.min_area or area > self.max_area:
            return 0.0, None

        aspect_ratio = float(w) / float(h)
        if aspect_ratio < self.min_ar or aspect_ratio > self.max_ar:
            return 0.0, None

        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        if extent < self.min_extent:
            return 0.0, None

        # Prefer lower-half of vehicle (where plates are mounted)
        H, W = frame_shape[:2]
        y_center = y + h / 2.0
        pos_weight = 1.2 if (y_center / H) > 0.45 else 0.8

        # Target ideal aspect ratio is ~4.0 (standard plate 500x120mm)
        ar_score = 1.0 - abs(aspect_ratio - 4.0) / 4.0
        ar_score = max(0.1, ar_score)

        score = (ar_score * 0.45 + extent * 0.35 + (area / self.max_area) * 0.20) * pos_weight
        return score, (x, y, w, h)

    def detect(self, original_image, candidate_mask):
        """
        Locate candidate plate contours, rank them, and extract the deskewed plate crop.
        
        Returns:
            dict containing:
            - 'plate_crop': Cropped and deskewed plate image (BGR)
            - 'bbox': [x, y, w, h] in original image
            - 'angle': Detected tilt angle
            - 'contour': Best contour
            - 'score': Confidence score
            - 'candidates': All evaluated candidate bounding boxes
        """
        contours, _ = cv2.findContours(
            candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        best_candidate = None
        best_score = -1.0

        for cnt in contours:
            score, bbox = self.score_contour(cnt, original_image.shape)
            if score > 0:
                rect = cv2.minAreaRect(cnt)
                candidates.append({
                    "score": score,
                    "bbox": bbox,
                    "rect": rect,
                    "contour": cnt
                })
                if score > best_score:
                    best_score = score
                    best_candidate = {
                        "score": score,
                        "bbox": bbox,
                        "rect": rect,
                        "contour": cnt
                    }

        # Fallback if no candidate passed strict filters:
        # Search for largest horizontal rectangle in the lower half of image
        if best_candidate is None:
            H, W = original_image.shape[:2]
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 80 and h > 20 and (float(w) / h) > 1.8 and y > (H * 0.3):
                    rect = cv2.minAreaRect(cnt)
                    best_candidate = {
                        "score": 0.5,
                        "bbox": (x, y, w, h),
                        "rect": rect,
                        "contour": cnt
                    }
                    break

        # If still none, use default central lower-bumper ROI
        if best_candidate is None:
            H, W = original_image.shape[:2]
            default_w, default_h = int(W * 0.4), int(H * 0.15)
            default_x = (W - default_w) // 2
            default_y = int(H * 0.65)
            plate_crop = original_image[default_y:default_y+default_h, default_x:default_x+default_w]
            return {
                "plate_crop": plate_crop,
                "bbox": [default_x, default_y, default_w, default_h],
                "angle": 0.0,
                "contour": None,
                "score": 0.2,
                "candidates": []
            }

        # Deskew using minAreaRect
        rect = best_candidate["rect"]
        rectified_plate, angle = self.deskew_plate(original_image, rect)
        
        # Verify rectified plate has valid dimensions
        if rectified_plate.shape[0] < 10 or rectified_plate.shape[1] < 30:
            x, y, w, h = best_candidate["bbox"]
            rectified_plate = original_image[y:y+h, x:x+w]
            angle = 0.0

        return {
            "plate_crop": rectified_plate,
            "bbox": best_candidate["bbox"],
            "angle": angle,
            "contour": best_candidate["contour"],
            "score": best_candidate["score"],
            "candidates": candidates
        }
