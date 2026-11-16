"""
Dataset Generator for Automatic Number Plate Recognition (ANPR)
Generates:
1. Isolated character training dataset (0-9, A-Z) with morphological augmentations.
2. Synthetic realistic vehicle surveillance scenes under diverse challenging conditions
   (Normal daylight, Low-light/Night, Tilted/Skewed, Weather/Sensor Noise, Glare).
"""

import os
import json
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

CLASSES = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

def get_system_font(font_name="arial.ttf", size=24):
    """Attempt to load system TrueType font, falling back to default."""
    windir = os.environ.get("WINDIR", "C:\\Windows")
    font_path = os.path.join(windir, "Fonts", font_name)
    if os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()

def generate_character_dataset(samples_per_class=50, img_size=(28, 28)):
    """
    Generate synthetic character dataset for 36 classes (0-9, A-Z)
    with morphological variations, rotations, scaling, and noise.
    
    Returns:
        X (np.ndarray): Shape (N, 28, 28), float32 normalized [0, 1]
        y (np.ndarray): Shape (N,), int64 class indices
        CLASSES (list): List of class string characters
    """
    font_names = ["arial.ttf", "calibri.ttf", "consola.ttf", "cour.ttf"]
    X = []
    y = []

    for class_idx, char in enumerate(CLASSES):
        for s in range(samples_per_class):
            # Select random font & size
            font_name = random.choice(font_names)
            font_size = random.randint(22, 28)
            font = get_system_font(font_name, font_size)

            # Draw character onto blank canvas
            canvas_size = (48, 48)
            pil_img = Image.new("L", canvas_size, color=0)
            draw = ImageDraw.Draw(pil_img)

            # Center calculation
            bbox = draw.textbbox((0, 0), char, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x_offset = (canvas_size[0] - w) // 2 - bbox[0] + random.randint(-2, 2)
            y_offset = (canvas_size[1] - h) // 2 - bbox[1] + random.randint(-2, 2)
            draw.text((x_offset, y_offset), char, fill=255, font=font)

            char_np = np.array(pil_img, dtype=np.uint8)

            # Apply geometric / morphological augmentations
            # 1. Rotation (-12 to +12 degrees)
            angle = random.uniform(-12, 12)
            M_rot = cv2.getRotationMatrix2D((canvas_size[0] // 2, canvas_size[1] // 2), angle, 1.0)
            char_np = cv2.warpAffine(char_np, M_rot, canvas_size, borderValue=0)

            # 2. Random Dilation or Erosion (stroke thickness variation)
            aug_choice = random.random()
            if aug_choice < 0.3:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                char_np = cv2.dilate(char_np, kernel, iterations=1)
            elif aug_choice < 0.5:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                char_np = cv2.erode(char_np, kernel, iterations=1)

            # 3. Add Gaussian or salt-pepper noise
            if random.random() < 0.4:
                noise = np.random.normal(0, random.uniform(5, 20), char_np.shape)
                char_np = np.clip(char_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # 4. Find bounding box of foreground character and crop with padding
            coords = cv2.findNonZero(char_np)
            if coords is not None:
                cx, cy, cw, ch = cv2.boundingRect(coords)
                cropped = char_np[cy:cy+ch, cx:cx+cw]
                # Pad to square
                max_side = max(cw, ch)
                square = np.zeros((max_side, max_side), dtype=np.uint8)
                px = (max_side - cw) // 2
                py = (max_side - ch) // 2
                square[py:py+ch, px:px+cw] = cropped
                final_char = cv2.resize(square, img_size, interpolation=cv2.INTER_AREA)
            else:
                final_char = cv2.resize(char_np, img_size, interpolation=cv2.INTER_AREA)

            # Normalize to [0, 1]
            final_norm = final_char.astype(np.float32) / 255.0
            X.append(final_norm)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, CLASSES


def create_license_plate_image(plate_text="DL01AB1234", plate_type="white", plate_size=(260, 65)):
    """
    Render a crisp, realistic standard vehicle license plate.
    Types: 'white' (private), 'yellow' (commercial), 'green' (EV)
    """
    w, h = plate_size
    bg_colors = {
        "white": (245, 245, 247),
        "yellow": (255, 204, 0),
        "green": (15, 120, 50)
    }
    fg_colors = {
        "white": (15, 15, 15),
        "yellow": (10, 10, 10),
        "green": (250, 250, 250)
    }

    bg_color = bg_colors.get(plate_type, (245, 245, 247))
    fg_color = fg_colors.get(plate_type, (15, 15, 15))

    img = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # 1. Outer & Inner Border
    draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(30, 30, 30), width=2)
    draw.rectangle([(4, 4), (w - 5, h - 5)], outline=(80, 80, 80), width=1)

    # 2. Blue strip on the left (HSRP / EU standard style)
    strip_w = int(w * 0.11)
    draw.rectangle([(5, 5), (strip_w, h - 5)], fill=(0, 51, 153))
    
    # Text "IND" inside strip
    strip_font = get_system_font("arial.ttf", size=9)
    draw.text((8, h - 20), "IND", fill=(255, 255, 255), font=strip_font)
    
    # Chakra/circle marker
    draw.ellipse([(10, 12), (strip_w - 6, 26)], outline=(255, 204, 0), width=1)

    # 3. Main Alphanumeric Registration Text
    text_font = get_system_font("arial.ttf", size=32)
    # Check text bbox
    text_bbox = draw.textbbox((0, 0), plate_text, font=text_font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]

    # Center text in the remaining plate area
    available_w = w - strip_w - 10
    start_x = strip_w + 5 + (available_w - tw) // 2
    start_y = (h - th) // 2 - text_bbox[1]

    # Draw embossed-style text
    draw.text((start_x, start_y), plate_text, fill=fg_color, font=text_font)

    # Convert to OpenCV BGR
    plate_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return plate_bgr


def generate_vehicle_scene(
    plate_text="DL01AB1234",
    condition="normal",
    tilt_angle=0.0,
    car_color=(40, 50, 120),
    scene_size=(480, 640)  # height, width
):
    """
    Synthesize a surveillance camera view of a vehicle at a parking gate.
    
    Args:
        plate_text (str): Alphanumeric registration string
        condition (str): 'normal', 'low_light', 'tilted', 'noisy', 'glare'
        tilt_angle (float): Skew angle in degrees
        car_color (tuple): BGR color of vehicle body
        scene_size (tuple): (H, W) of the surveillance frame
        
    Returns:
        scene (np.ndarray): 3-channel BGR image
        metadata (dict): Ground truth bounding box and text
    """
    H, W = scene_size
    scene = np.zeros((H, W, 3), dtype=np.uint8)

    # 1. Background (Pavement / Gate Wall)
    # Gradient pavement
    for y in range(H):
        gray_val = int(80 + (y / H) * 45)
        scene[y, :] = (gray_val - 10, gray_val, gray_val + 5)

    # Add asphalt texture noise
    asphalt_noise = np.random.randint(-15, 15, (H, W, 3), dtype=np.int16)
    scene = np.clip(scene.astype(np.int16) + asphalt_noise, 0, 255).astype(np.uint8)

    # 2. Draw Vehicle Front Grille and Bumper
    car_top = int(H * 0.18)
    car_bottom = int(H * 0.88)
    car_left = int(W * 0.08)
    car_right = int(W * 0.92)

    # Draw car hood / bumper curved polygon
    bumper_pts = np.array([
        [car_left + 40, car_top],
        [car_right - 40, car_top],
        [car_right, car_top + 100],
        [car_right - 10, car_bottom],
        [car_left + 10, car_bottom],
        [car_left, car_top + 100]
    ], np.int32)
    cv2.fillPoly(scene, [bumper_pts], car_color)
    cv2.polylines(scene, [bumper_pts], isClosed=True, color=(20, 20, 20), thickness=3)

    # Radiator Grille (Mesh pattern)
    grille_top = car_top + 50
    grille_bottom = grille_top + 110
    grille_left = car_left + 80
    grille_right = car_right - 80
    cv2.rectangle(scene, (grille_left, grille_top), (grille_right, grille_bottom), (25, 25, 25), -1)
    for gy in range(grille_top + 8, grille_bottom, 12):
        cv2.line(scene, (grille_left + 5, gy), (grille_right - 5, gy), (55, 55, 55), 2)

    # Headlights (Left & Right)
    cv2.ellipse(scene, (car_left + 45, car_top + 60), (35, 20), -15, 0, 360, (230, 240, 250), -1)
    cv2.ellipse(scene, (car_right - 45, car_top + 60), (35, 20), 15, 0, 360, (230, 240, 250), -1)
    cv2.ellipse(scene, (car_left + 45, car_top + 60), (35, 20), -15, 0, 360, (100, 100, 100), 2)
    cv2.ellipse(scene, (car_right - 45, car_top + 60), (35, 20), 15, 0, 360, (100, 100, 100), 2)

    # 3. Create License Plate and Mount Holder
    plate_w = 230
    plate_h = 58
    plate_img = create_license_plate_image(plate_text=plate_text, plate_size=(plate_w, plate_h))

    # Center position for plate
    plate_cx = W // 2
    plate_cy = car_bottom - 75

    # Black bumper plate bracket
    holder_pad = 8
    hx1 = plate_cx - plate_w // 2 - holder_pad
    hy1 = plate_cy - plate_h // 2 - holder_pad
    hx2 = plate_cx + plate_w // 2 + holder_pad
    hy2 = plate_cy + plate_h // 2 + holder_pad
    cv2.rectangle(scene, (hx1, hy1), (hx2, hy2), (18, 18, 18), -1)
    cv2.rectangle(scene, (hx1, hy1), (hx2, hy2), (50, 50, 50), 2)

    # 4. Embed Plate (with potential tilt)
    px1 = plate_cx - plate_w // 2
    py1 = plate_cy - plate_h // 2
    px2 = px1 + plate_w
    py2 = py1 + plate_h

    # Base bbox without tilt
    bbox = [px1, py1, plate_w, plate_h]

    # If tilt is requested
    if abs(tilt_angle) > 0.1:
        M = cv2.getRotationMatrix2D((plate_w / 2, plate_h / 2), tilt_angle, 1.0)
        tilted_plate = cv2.warpAffine(plate_img, M, (plate_w, plate_h), borderValue=(18, 18, 18))
        scene[py1:py2, px1:px2] = tilted_plate
    else:
        scene[py1:py2, px1:px2] = plate_img

    # 5. Apply Environmental / Sensor Condition Effects
    if condition == "low_light":
        # Simulates night-time surveillance
        # Darken scene drastically
        scene = (scene.astype(np.float32) * 0.28).astype(np.uint8)
        # Headlights on glow effect
        glow_left = np.zeros_like(scene, dtype=np.float32)
        glow_right = np.zeros_like(scene, dtype=np.float32)
        cv2.circle(glow_left, (car_left + 45, car_top + 60), 80, (180, 210, 240), -1)
        cv2.circle(glow_right, (car_right - 45, car_top + 60), 80, (180, 210, 240), -1)
        glow = cv2.GaussianBlur(glow_left + glow_right, (65, 65), 0)
        scene = np.clip(scene.astype(np.float32) + glow * 0.7, 0, 255).astype(np.uint8)
        # Add high-ISO Poisson/Gaussian sensor noise
        noise = np.random.normal(0, 14, scene.shape).astype(np.int16)
        scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif condition == "glare":
        # Sunlight glare reflection near plate
        glare_mask = np.zeros((H, W), dtype=np.float32)
        cv2.circle(glare_mask, (px1 + 40, py1 + 10), 60, 1.0, -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (55, 55), 0)[:, :, np.newaxis]
        scene = np.clip(scene.astype(np.float32) + glare_mask * 130, 0, 255).astype(np.uint8)

    elif condition == "noisy":
        # Atmospheric dust & sensor noise
        gauss = np.random.normal(0, 18, scene.shape).astype(np.int16)
        scene = np.clip(scene.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
        # Blur slightly (motion/lens blur)
        scene = cv2.GaussianBlur(scene, (3, 3), 0)

    metadata = {
        "plate_text": plate_text,
        "bbox": bbox,
        "tilt_angle": tilt_angle,
        "condition": condition
    }

    return scene, metadata


def generate_benchmark_dataset(output_dir="data/raw", num_samples=25):
    """
    Generate a diverse benchmark dataset across conditions and save to disk with manifest.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sample_plates = [
        ("DL01AB1234", "normal", 0.0),
        ("MH12DE4567", "normal", 0.0),
        ("KA05MJ9876", "normal", 0.0),
        ("TN07BK3321", "normal", 0.0),
        ("TS09FA2024", "normal", 0.0),
        # Low light / Night conditions
        ("DL08CZ5544", "low_light", 0.0),
        ("HR26DK8899", "low_light", 0.0),
        ("UP16AF1122", "low_light", 0.0),
        ("MH04GH7766", "low_light", 0.0),
        ("KA01ER3456", "low_light", 0.0),
        # Tilted / Skewed conditions
        ("GJ01XY9012", "tilted", 12.0),
        ("RJ14CD4321", "tilted", -15.0),
        ("KL07BT6543", "tilted", 18.0),
        ("WB02AC7890", "tilted", -10.0),
        ("AP09PK5678", "tilted", 14.0),
        # Noisy / Weather conditions
        ("CH01TC1111", "noisy", 0.0),
        ("MP09ZA9988", "noisy", 0.0),
        ("GA03X7777", "noisy", 0.0),
        ("PB10BW2233", "noisy", 0.0),
        ("OD02MK4455", "noisy", 0.0),
        # Glare / Reflections
        ("DL03CC8811", "glare", 0.0),
        ("MH02BY3344", "glare", 0.0),
        ("KA03NF5566", "glare", 0.0),
        ("TN09EF7788", "glare", 0.0),
        ("HR51AM9900", "glare", 0.0),
    ]

    manifest = []
    car_colors = [
        (40, 50, 140),   # Navy blue
        (130, 30, 30),   # Crimson
        (35, 120, 50),   # Forest green
        (180, 180, 180), # Silver
        (30, 30, 30),    # Midnight black
        (220, 220, 230), # Pearl white
    ]

    for idx, (text, cond, tilt) in enumerate(sample_plates[:num_samples]):
        color = random.choice(car_colors)
        scene, meta = generate_vehicle_scene(
            plate_text=text,
            condition=cond,
            tilt_angle=tilt,
            car_color=color
        )
        filename = f"vehicle_{idx+1:03d}_{cond}_{text}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, scene)
        
        meta["image_path"] = filepath
        meta["filename"] = filename
        manifest.append(meta)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Generated {len(manifest)} benchmark vehicle scenes in '{output_dir}'.")
    return manifest

if __name__ == "__main__":
    print("Generating character dataset...")
    Xc, yc, classes = generate_character_dataset(samples_per_class=10)
    print(f"Generated characters shape: {Xc.shape}, classes: {len(classes)}")
    print("Generating vehicle scenes...")
    generate_benchmark_dataset()
