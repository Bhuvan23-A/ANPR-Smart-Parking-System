"""
End-to-End Automatic Number Plate Recognition (ANPR) Pipeline
Coordinates the complete Pattern Recognition workflow:
Surveillance Image -> Preprocessing -> Plate Localization & Deskewing ->
Character Segmentation -> HOG-SVM/MLP Recognition -> Positional Syntax Disambiguation ->
Database Query & Smart Parking Gate Automation.
"""

import cv2
import numpy as np

from src.preprocessing import preprocess_pipeline
from src.plate_detector import PlateDetector
from src.character_segmenter import CharacterSegmenter
from src.ocr_engine import OCREngine
from src.postprocessor import (
    clean_raw_ocr_string,
    disambiguate_license_plate,
    match_registered_database
)
from src.parking_system import SmartParkingSystem

class ANPRPipeline:
    def __init__(self, model_type="svm", db_path="data/parking.db", models_dir="data/models"):
        self.detector = PlateDetector()
        self.segmenter = CharacterSegmenter()
        self.ocr = OCREngine(model_type=model_type, models_dir=models_dir)
        self.parking = SmartParkingSystem(db_path=db_path)
        
        # Ensure OCR model is loaded
        self.ocr.load_model(model_type)

    def process(self, image_input, action="entry", mall_spent=0.0, simulated_duration_mins=None):
        """
        Run the complete pipeline on an input image.
        
        Args:
            image_input: Filepath (str) or BGR numpy array
            action (str): 'entry', 'exit', or 'detect_only'
            mall_spent (float): Mall shopping expenditure for exit discount
            simulated_duration_mins (float): Simulated parking duration for demo exit
            
        Returns:
            dict containing every intermediate representation, extracted text, and gate response.
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not read image from {image_input}")
        else:
            image = image_input.copy()

        # Step 1: Preprocessing & Contrast Enhancement
        prep_data = preprocess_pipeline(image)

        # Step 2: License Plate Localization & Affine Deskewing
        det_data = self.detector.detect(image, prep_data["candidate_mask"])
        plate_crop = det_data["plate_crop"]

        # Step 3: Character Segmentation
        segmented_chars, seg_debug = self.segmenter.segment(plate_crop)

        # Step 4: Character Recognition (OCR)
        raw_text, ocr_conf, char_details = self.ocr.recognize_plate(segmented_chars)

        # Step 5: Post-Processing & Positional Disambiguation
        cleaned_text = clean_raw_ocr_string(raw_text)
        disambiguated_text = disambiguate_license_plate(cleaned_text)

        # Fuzzy match against registered VIP / subscriber database
        registered_plates = self.parking.get_registered_plates()
        final_plate, dist, is_exact = match_registered_database(
            disambiguated_text, registered_plates, max_distance=1
        )

        # Step 6: Smart Parking Action
        parking_result = None
        if action == "entry":
            parking_result = self.parking.process_entry(final_plate)
        elif action == "exit":
            parking_result = self.parking.process_exit(
                final_plate,
                mall_spent=mall_spent,
                simulated_duration_mins=simulated_duration_mins
            )

        # Annotated image for display
        annotated = image.copy()
        bx, by, bw, bh = det_data["bbox"]
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
        label = f"{final_plate} ({ocr_conf*100:.1f}%)"
        cv2.putText(annotated, label, (bx, max(25, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return {
            "original_image": image,
            "preprocessing": prep_data,
            "detection": det_data,
            "plate_crop": plate_crop,
            "segmentation": segmented_chars,
            "seg_debug": seg_debug,
            "raw_text": raw_text,
            "ocr_confidence": ocr_conf,
            "char_details": char_details,
            "disambiguated_text": disambiguated_text,
            "final_plate": final_plate,
            "matched_registered": final_plate in registered_plates,
            "parking_result": parking_result,
            "annotated_image": annotated
        }
