"""
Automatic Number Plate Recognition (ANPR) for Smart Parking System
Main CLI Driver and Demonstration Script

Demonstrates:
1. Surveillance Camera vehicle frame ingestion
2. Multi-stage Pattern Recognition (Preprocessing -> Localization -> Segmentation -> HOG-SVM Recognition)
3. Smart Parking Entry Gate (Slot Allocation, Ticket issuance, Barrier OPEN)
4. Smart Parking Exit Gate (Duration tracking, Mall spending discount, FASTag payment receipt)
"""

import os
import sys
import argparse
import json
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import ANPRPipeline
from src.parking_system import SmartParkingSystem
from src.dataset_generator import generate_benchmark_dataset

def print_banner():
    banner = """
================================================================================
   AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) FOR SMART PARKING MANAGEMENT
                    Pattern Recognition Course Project
================================================================================
    """
    print(banner)

def demo_single_vehicle(image_path, model_type="svm", mall_spent=650.0, simulated_duration_mins=145.0):
    print(f"\n[+] Loading ANPR Pipeline with [{model_type.upper()}] Character Classifier...")
    pipeline = ANPRPipeline(model_type=model_type)

    print(f"[+] Processing Surveillance Image: {image_path}")
    
    # 1. Simulate Entry Gate
    print("\n" + "=" * 50)
    print("           STAGE 1: PARKING ENTRY GATE")
    print("=" * 50)
    t0 = time.time()
    res_entry = pipeline.process(image_path, action="entry")
    entry_latency = (time.time() - t0) * 1000.0

    print(f"[*] Plate Detected BBox     : {res_entry['detection']['bbox']}")
    print(f"[*] Detected Tilt Angle     : {res_entry['detection']['angle']:.2f} degrees")
    print(f"[*] Segmented Characters    : {len(res_entry['segmentation'])} glyphs")
    print(f"[*] Raw OCR Readout         : {res_entry['raw_text']}")
    print(f"[*] Syntactic Disambiguated : {res_entry['disambiguated_text']}")
    print(f"[*] Final Verified Plate    : {res_entry['final_plate']} (Confidence: {res_entry['ocr_confidence']*100:.1f}%)")
    print(f"[*] Processing Pipeline Time: {entry_latency:.2f} ms")

    entry_ticket = res_entry["parking_result"]
    print("\n--- ISSUED SMART ENTRY TICKET ---")
    if entry_ticket.get("success"):
        print(f"  Ticket Status    : VALID ENTRY")
        print(f"  Vehicle Number   : {entry_ticket['plate_number']}")
        print(f"  Owner / Category : {entry_ticket['owner_name']} ({entry_ticket['category']})")
        print(f"  Assigned Slot    : {entry_ticket['slot_id']}")
        print(f"  Timestamp        : {entry_ticket['entry_time']}")
        print(f"  Gate Barrier     : [{entry_ticket['barrier']}]")
        print(f"  Message          : {entry_ticket['message']}")
    else:
        print(f"  Notice: {entry_ticket.get('message')}")

    # 2. Simulate Exit Gate & Automated Billing
    print("\n" + "=" * 50)
    print("       STAGE 2: PARKING EXIT GATE & AUTOMATED BILLING")
    print("=" * 50)
    print(f"[*] Simulating Vehicle Departure after {simulated_duration_mins:.0f} minutes...")
    print(f"[*] Customer Mall Purchase Receipt : ₹{mall_spent:.2f} (Eligible for Mall Parking Discount)")

    t0 = time.time()
    res_exit = pipeline.process(
        image_path,
        action="exit",
        mall_spent=mall_spent,
        simulated_duration_mins=simulated_duration_mins
    )
    exit_latency = (time.time() - t0) * 1000.0

    exit_receipt = res_exit["parking_result"]
    if exit_receipt.get("success"):
        print("\n==================================================")
        print("            METROPOLIS MALL PARKING RECEIPT       ")
        print("==================================================")
        print(f"  Receipt ID       : REC-{exit_receipt['log_id']:05d}")
        print(f"  Vehicle Number   : {exit_receipt['plate_number']}")
        print(f"  Permit Category  : {exit_receipt['category']}")
        print(f"  Parked Slot      : {exit_receipt['slot_id']}")
        print(f"  Entry Time       : {exit_receipt['entry_time']}")
        print(f"  Exit Time        : {exit_receipt['exit_time']}")
        print(f"  Duration Parked  : {exit_receipt['duration_formatted']} ({exit_receipt['duration_minutes']} mins)")
        print(f"  ----------------------------------------------")
        print(f"  Base Parking Fee : ₹{exit_receipt['base_fee']:.2f}")
        print(f"  Applied Discount : -₹{exit_receipt['discount_amount']:.2f}")
        print(f"  TOTAL PAYABLE    : ₹{exit_receipt['final_amount']:.2f}")
        print(f"  Payment Mode     : {exit_receipt['payment_method']} (Electronic Auto-Deduct)")
        print(f"  Payment Status   : {exit_receipt['payment_status']}")
        print(f"  Exit Barrier     : [{exit_receipt['barrier']}]")
        print("==================================================")
        print(f"  {exit_receipt['message']}")
        print("==================================================\n")
    else:
        print(f"  Exit Failed: {exit_receipt.get('message')}")

def run_batch_demo():
    print("\n[+] Running Batch Demonstration on Raw Vehicle Dataset...")
    manifest_path = "data/raw/manifest.json"
    if not os.path.exists(manifest_path):
        print("[!] Dataset manifest not found. Generating now...")
        generate_benchmark_dataset()

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    pipeline = ANPRPipeline(model_type="svm")
    print(f"\n{'IMAGE FILENAME':<38} | {'CONDITION':<10} | {'GROUND TRUTH':<11} | {'DETECTED':<11} | {'CONF':<6} | {'BARRIER'}")
    print("-" * 92)

    for item in manifest[:10]:
        res = pipeline.process(item["image_path"], action="entry")
        gt = item["plate_text"]
        pred = res["final_plate"]
        conf = f"{res['ocr_confidence']*100:.1f}%"
        barrier = res["parking_result"].get("barrier", "N/A") if res["parking_result"] else "N/A"
        cond = item["condition"]
        fname = item["filename"]
        print(f"{fname:<38} | {cond:<10} | {gt:<11} | {pred:<11} | {conf:<6} | [{barrier}]")

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="ANPR Smart Parking System")
    parser.add_argument("--image", type=str, default=None, help="Path to vehicle image")
    parser.add_argument("--model", type=str, default="svm", choices=["svm", "knn", "mlp", "template"], help="OCR model")
    parser.add_argument("--batch", action="store_true", help="Run batch test on dataset")
    parser.add_argument("--mall_spent", type=float, default=650.0, help="Customer mall spending in Rs")
    parser.add_argument("--duration", type=float, default=145.0, help="Simulated parking duration in mins")
    
    args = parser.parse_args()

    if args.batch:
        run_batch_demo()
    elif args.image:
        demo_single_vehicle(args.image, model_type=args.model, mall_spent=args.mall_spent, simulated_duration_mins=args.duration)
    else:
        # Default run with first sample in data/raw
        sample_img = "data/raw/vehicle_001_normal_DL01AB1234.png"
        if not os.path.exists(sample_img):
            generate_benchmark_dataset()
        demo_single_vehicle(sample_img, model_type=args.model, mall_spent=args.mall_spent, simulated_duration_mins=args.duration)

if __name__ == "__main__":
    main()
