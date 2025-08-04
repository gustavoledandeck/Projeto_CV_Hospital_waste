#!/usr/bin/env python3
import pathlib
import shutil
import random
import cv2
import numpy as np
import os
from tqdm import tqdm

# --- Configuration ---
# Path to the raw data captured by record_class.py
RAW_DATA_DIR = pathlib.Path("data/raw")
# Path to the directory where the processed dataset will be stored
PROCESSED_DATA_DIR = pathlib.Path("data/processed")
# Train/validation split ratio
TRAIN_VAL_SPLIT = 0.8  # 80% for training, 20% for validation
# List of classes (should match the folder names in data/raw)
CLASSES = ["low", "medium", "high", "normal"]
# Add padding around the detected object when cropping (as a percentage)
PADDING = 0.10 

def crop_object_from_image(img_path):
    """
    This function loads an image, finds the largest object using contour detection,
    and returns a cropped version of that object.
    """
    frame = cv2.imread(str(img_path))
    if frame is None:
        return None

    # Preprocessing for detection (same as in inference script)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 30, 150)

    # Find the largest contour
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None # No object found

    # Get the largest contour by area
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)

    # Apply padding to the bounding box
    pad_w = int(w * PADDING)
    pad_h = int(h * PADDING)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame.shape[1], x + w + pad_w)
    y2 = min(frame.shape[0], y + h + pad_h)

    # Crop the image to the padded bounding box
    cropped_img = frame[y1:y2, x1:x2]
    return cropped_img

# --- Main Function ---
def prepare_dataset():
    """
    This script organizes the raw images into a training and validation set.
    It now automatically crops the main object from each image before saving.
    """
    
    # --- Cleanup and Setup ---
    if PROCESSED_DATA_DIR.exists():
        print(f"Removing existing processed data directory: {PROCESSED_DATA_DIR}")
        shutil.rmtree(PROCESSED_DATA_DIR)

    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    train_dir = PROCESSED_DATA_DIR / "train"
    val_dir = PROCESSED_DATA_DIR / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    print(f"Preparing dataset in: {PROCESSED_DATA_DIR}\n")

    # --- Process each class ---
    for cls in CLASSES:
        print(f"Processing class: {cls}")
        
        (train_dir / cls).mkdir()
        (val_dir / cls).mkdir()
        
        raw_class_dir = RAW_DATA_DIR / cls
        if not raw_class_dir.exists():
            print(f"Warning: Raw data directory for class '{cls}' not found. Skipping.")
            continue
            
        image_files = list(raw_class_dir.glob("*.jpg"))
        random.shuffle(image_files)
        
        split_idx = int(len(image_files) * TRAIN_VAL_SPLIT)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        print(f"  - {len(train_files)} images for training")
        print(f"  - {len(val_files)} images for validation")
        
        # --- Crop and Copy files ---
        for f in tqdm(train_files, desc=f"  Cropping train '{cls}'"):
            cropped = crop_object_from_image(f)
            if cropped is not None:
                cv2.imwrite(str(train_dir / cls / f.name), cropped)
            
        for f in tqdm(val_files, desc=f"  Cropping val '{cls}'"):
            cropped = crop_object_from_image(f)
            if cropped is not None:
                cv2.imwrite(str(val_dir / cls / f.name), cropped)
            
    print("\n✅  Dataset preparation complete!")

if __name__ == "__main__":
    prepare_dataset()
