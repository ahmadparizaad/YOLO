import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import re
import os
from collections import Counter

# Load YOLO model
yolo_model = YOLO("best.pt")

CLASSES = [
    "title_block", "filter_table", "fan_motor_table", "coil_table",
    "drain_table", "damper_table", "elevation_view", "plan_view",
]

def remove_blue_hatching(img):
    if img is None or img.size == 0: return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = img.copy()
    result[blue_mask > 0] = [255, 255, 255]
    return result

def rotate_for_vertical_text(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def test_psm_on_roi(img, rotate=False, remove_hatching=False):
    if img is None or img.size == 0: return {}
    
    if remove_hatching: img = remove_blue_hatching(img)
    if rotate: img = rotate_for_vertical_text(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 6
    gray_scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    
    # Sharpen
    blurred = cv2.GaussianBlur(gray_scaled, (0, 0), 3)
    gray_scaled = cv2.addWeighted(gray_scaled, 2.0, blurred, -1.0, 0)
    
    # Otsu only (as per optimization)
    _, binary = cv2.threshold(gray_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.copyMakeBorder(binary, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    psm_results = {}
    # Testing a wide range of PSM modes
    for psm in [3, 4, 6, 7, 8, 11, 13]:
        config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(binary, config=config)
        nums = [int(x) for x in re.findall(r"\d{2,5}", text)]
        psm_results[psm] = nums
        
    return psm_results

def run_test():
    image_folder = "images"
    # Test on a subset of images
    test_images = [
        "1f30e6cc-AHU-GF-06.png",
        "17a20cc6-AHU-GF-08.png",
        "3c616709-AHU-GF-15.png",
        "7b9e7db4-AHU-GF-03.png",
        "ded8dfc5-AHU-GF-02.png"
    ]
    
    overall_stats = {psm: 0 for psm in [3, 4, 6, 7, 8, 11, 13]}
    
    for img_name in test_images:
        img_path = os.path.join(image_folder, img_name)
        image = cv2.imread(img_path)
        if image is None: continue
        
        results = yolo_model(image, verbose=False)
        elevation_box = None
        plan_box = None
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if CLASSES[cls_id] == "elevation_view":
                    elevation_box = box.xyxy[0].tolist()
                elif CLASSES[cls_id] == "plan_view":
                    plan_box = box.xyxy[0].tolist()
        
        if not elevation_box or not plan_box: continue
        
        # Extract ROIs (simplified logic from pipeline)
        ex1, ey1, ex2, ey2 = map(int, elevation_box)
        ew, eh = ex2 - ex1, ey2 - ey1
        img_h, img_w = image.shape[:2]
        
        length_roi = image[int(ey1 + 0.75*eh):min(int(ey2 + 50), img_h), int(ex1 + 0.10*ew):int(ex1 + 0.90*ew)]
        height_roi = image[int(ey1):int(ey1 + 0.65*eh), max(int(ex2 - 200), 0):min(int(ex2 + 400), img_w)]
        
        px1, py1, px2, py2 = map(int, plan_box)
        pw, ph = px2 - px1, py2 - py1
        width_roi = image[int(py1 + 0.10*ph):int(py1 + 0.90*ph), max(int(px2 - 200), 0):min(int(px2 + 400), img_w)]
        
        rois = [
            ("length", length_roi, False, True),
            ("height", height_roi, True, False),
            ("width", width_roi, True, False)
        ]
        
        print(f"\n--- Testing Image: {img_name} ---")
        for name, roi, rotate, hatching in rois:
            results = test_psm_on_roi(roi, rotate, hatching)
            print(f"  ROI: {name}")
            for psm, nums in results.items():
                # A "success" is finding at least one plausible dimension (1000-6000)
                plausible = [n for n in nums if 1000 <= n <= 6000]
                if plausible:
                    overall_stats[psm] += 1
                    print(f"    PSM {psm}: Found {plausible}")
                else:
                    print(f"    PSM {psm}: No plausible numbers")

    print("\n=== OVERALL PSM SUCCESS RATES (Plausible Dimensions Found) ===")
    for psm, count in sorted(overall_stats.items(), key=lambda x: -x[1]):
        print(f"PSM {psm}: {count} successes")

if __name__ == "__main__":
    run_test()
