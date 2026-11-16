"""
Image Preprocessing Module for Automatic Number Plate Recognition
Handles:
1. Grayscale conversion
2. Noise reduction using edge-preserving Bilateral Filtering
3. Contrast-Limited Adaptive Histogram Equalization (CLAHE) for low-light & shadows
4. Morphological Top-hat / Black-hat transforms
5. Vertical Sobel / Canny edge filtering
6. Morphological Closing to form candidate plate blobs
"""

import cv2
import numpy as np

def convert_to_grayscale(image):
    """Convert BGR image to grayscale if not already."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()

def apply_clahe(gray_img, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Prevents noise amplification while significantly boosting local contrast
    in dark, night-time, or heavily shadowed vehicle images.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)

def bilateral_noise_filter(gray_img, d=9, sigma_color=75, sigma_space=75):
    """
    Apply Bilateral Filter to smooth noise while preserving sharp boundaries
    (crucial for plate edges and character strokes).
    """
    return cv2.bilateralFilter(gray_img, d, sigma_color, sigma_space)

def morphological_gradient_plate_highlight(gray_img):
    """
    Apply Top-hat and Black-hat morphological transforms to enhance
    textural differences between the plate background and alphanumeric characters.
    Top-hat = src - open(src) (bright regions on dark background)
    Black-hat = close(src) - src (dark regions on bright background)
    """
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    top_hat = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, rect_kernel)
    black_hat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, rect_kernel)
    
    # Combined gradient
    enhanced = cv2.add(gray_img, top_hat)
    enhanced = cv2.subtract(enhanced, black_hat)
    return enhanced

def detect_vertical_edges(gray_img):
    """
    Detect vertical edges using Sobel operator.
    License plates contain high frequencies of vertical character edges.
    """
    sobel_x = cv2.Sobel(gray_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    sobel_x = np.absolute(sobel_x)
    
    # Scale to 0-255
    min_val, max_val = np.min(sobel_x), np.max(sobel_x)
    if max_val > min_val:
        sobel_x = 255 * ((sobel_x - min_val) / (max_val - min_val))
    sobel_x = sobel_x.astype(np.uint8)
    return sobel_x

def morphological_close_to_blob(edge_img, kernel_size=(19, 5)):
    """
    Apply Otsu thresholding and morphological closing to merge discrete
    vertical character edges into a continuous rectangular candidate license plate mask.
    """
    # Otsu thresholding
    _, thresh = cv2.threshold(edge_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Horizontal morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Clean noise with erosion and dilation
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.erode(closed, kernel_clean, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel_clean, iterations=2)
    return cleaned

def preprocess_pipeline(image, is_low_light=None):
    """
    Full end-to-end preprocessing pipeline.
    
    Returns:
        dict containing intermediate stages and the final candidate edge blob mask:
        - 'gray'
        - 'clahe'
        - 'filtered'
        - 'sobel'
        - 'candidate_mask'
    """
    gray = convert_to_grayscale(image)
    
    # Automatic low-light check if not explicitly set
    mean_brightness = np.mean(gray)
    if is_low_light is None:
        is_low_light = mean_brightness < 70
        
    # Contrast Enhancement (CLAHE applied aggressively if low-light)
    clip = 4.0 if is_low_light else 2.5
    clahe_img = apply_clahe(gray, clip_limit=clip)
    
    # Bilateral smoothing
    filtered = bilateral_noise_filter(clahe_img)
    
    # Edge detection
    sobel = detect_vertical_edges(filtered)
    
    # Morphological candidate blob aggregation
    candidate_mask = morphological_close_to_blob(sobel)
    
    return {
        "gray": gray,
        "clahe": clahe_img,
        "filtered": filtered,
        "sobel": sobel,
        "candidate_mask": candidate_mask,
        "is_low_light": is_low_light,
        "mean_brightness": mean_brightness
    }
