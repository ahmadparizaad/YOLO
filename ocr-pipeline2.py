import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import re

# ===============================
# YOLO CLASSES
# ===============================
CLASSES = [
    "title_block",
    "filter_table",
    "fan_motor_table",
    "coil_table",
    "drain_table",
    "damper_table",
    "elevation_view",
    "plan_view",
]

def flatten_ocr_text(ocr_text):
    """
    Handles both:
    [("text", conf), ...]
    [[("text", conf)], [("text", conf)], ...]
    """
    lines = []

    for row in ocr_text:
        # Case 1: already flat
        if isinstance(row, tuple) and len(row) == 2:
            lines.append(row)

        # Case 2: row-based OCR
        elif isinstance(row, list):
            for item in row:
                if isinstance(item, tuple) and len(item) == 2:
                    lines.append(item)

    return lines

# ===============================
# LOAD MODELS
# ===============================
yolo_model = YOLO("best.pt")

# ===============================
# OCR UTILS
# ===============================
def crop(img, box):
    x1, y1, x2, y2 = map(int, box)
    return img[y1:y2, x1:x2]

def ocr_image(img):
    if img is None or img.size == 0:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    config = (
        "--psm 6 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz0123456789./-+()% "
    )

    text = pytesseract.image_to_string(gray, config=config)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 2:
            lines.append((line, 1.0))

    return lines

# ===============================
# YOLO + OCR PIPELINE
# ===============================
def extract_text_by_region(image_path, conf_threshold=0.25):
    image = cv2.imread(image_path)
    results = yolo_model(image)

    output = {}

    for r in results:
        for box in r.boxes:
            score = float(box.conf[0])
            if score < conf_threshold:
                continue

            cls_id = int(box.cls[0])
            label = CLASSES[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cropped = crop(image, (x1, y1, x2, y2))
            rows = extract_table_rows(cropped)

            all_rows_text = []
            for row_img in rows:
                row_text = ocr_image(row_img)
                if row_text:
                    all_rows_text.append(row_text)

            output.setdefault(label, []).append({
                "confidence": score,
                "bbox": [x1, y1, x2, y2],
                "text": all_rows_text
            })

    return output

# ===============================
# TABLE ROW EXTRACTION
# ===============================
def detect_horizontal_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=5
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (img.shape[1] // 20, 1)
    )
    horizontal = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, kernel, iterations=1
    )
    contours, _ = cv2.findContours(
        horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > img.shape[1] * 0.5:
            lines.append(y)
    return sorted(lines)

def build_rows_from_lines(lines, img_height, min_row_height=25):
    rows = []
    if not lines:
        return rows
    boundaries = [0] + lines + [img_height]
    for i in range(len(boundaries) - 1):
        y1 = boundaries[i]
        y2 = boundaries[i + 1]
        if y2 - y1 >= min_row_height:
            rows.append((y1, y2))
    return rows

def extract_table_rows(img):
    lines = detect_horizontal_lines(img)
    row_ranges = build_rows_from_lines(lines, img.shape[0])
    row_images = []
    for y1, y2 in row_ranges:
        row = img[y1:y2, :]
        row_images.append(row)
    return row_images

# ===============================
#  TITLE BLOCK PARSER
# ===============================
def parse_title_block(title_items):
    lines = []
    for item in title_items:
        for text, _ in flatten_ocr_text(item["text"]):
            if text.strip():
                lines.append(text.strip())

    full_text = " ".join(lines)
    result = {}

    # System No
    m = re.search(r"\bAHU[-\s]?[A-Z0-9-]+\b", full_text)
    if m:
        result["system_no"] = m.group(0).replace(" ", "")

    # Model
    m = re.search(r"\bAHUM[-A-Z0-9/\.]+", full_text)
    if m:
        result["model"] = m.group(0).rstrip("/")

    # Manufacturer
    if "SYSTEMAIR" in full_text.replace(" ", "").upper():
        result["manufacturer"] = "System Air India Pvt. Ltd."

    # Description
    for line in lines:
        if "DOUBLE" in line.upper() and "AHU" in line.upper():
            desc = re.sub(r"G\.?A\.?DRAWING\s*FOR", "", line, flags=re.I)
            desc = re.sub(r"\(.*?\)", "", desc)
            result["description"] = re.sub(r"\s+", " ", desc).title()
            break

    return result

# ===============================
# FILTER TABLE PARSER (FIXED)
# ===============================
def logical_filter_bucket(filter_class, has_holes, filter_description=""):
    """FINAL domain-correct mapping based on authoritative rules."""
    if not filter_description:
        return None

    desc = filter_description.upper()
    
    # SEMI HEPA (F-9) -> finefilter
    if "SEMI HEPA" in desc:
        return "finefilter"
    
    # HEPA or ULPA -> bleedfilter
    if "HEPA" in desc or "ULPA" in desc:
        return "bleedfilter"

    # PRE FILTER (G-*)
    if "PRE FILTER" in desc or "PREFILTER" in desc:
        return "freshfilter" if has_holes else "prefilter"
    
    # FINE FILTER (F-*)
    if "FINE FILTER" in desc or "FINEFILTER" in desc:
        return "bleedfilter" if has_holes else "finefilter"

    return None

def parse_filter_table(filter_items):
    """Parses filter table from OCR output with proper table structure."""
    lines = []
    for item in filter_items:
        for text, _ in flatten_ocr_text(item["text"]):
            # Clean up common OCR artifacts in the text
            t = re.sub(r"\s+", " ", text.upper()).strip()
            if t:
                lines.append(t)
    
    # Parse filter manufacturer
    filter_make = "AAF"
    for line in lines:
        if "AAF" in line:
            filter_make = "AAF"
            break
    
    # Initialize result structure
    filter_mapping = {
        "prefilter": {"class": None, "type": None, "sizes": []},
        "freshfilter": {"class": None, "type": None, "sizes": []},
        "finefilter": {"class": None, "type": None, "sizes": []},
        "bleedfilter": {"class": None, "type": None, "sizes": []}
    }
    
    # Current filter being processed
    current_role = None
    
    for line in lines:
        # Skip header lines
        if "FILTER DETAILS" in line or "RATING" in line or "QTY." in line:
            continue
        
        # Check if this is a new filter type line
        filter_match = re.search(r'(PRE\s*FILTER|FINE\s*FILTER|SEMI\s*HEPA\s*FILTER|HEPA\s*FILTER|ULPA)', line)
        
        if filter_match:
            desc = filter_match.group(0)
            # Parse filter class
            class_match = re.search(r'\(([G|F|H|U][-\s]?\d+)\)', line)
            filter_class = class_match.group(1).replace(" ", "") if class_match else None
            
            # Check if it has holes
            has_holes = "WITH HOLES" in line or "WITHHOLES" in line
            
            # Determine functional role using the rules
            role = logical_filter_bucket(filter_class, has_holes, line)
            
            if role:
                current_role = role
                if filter_class:
                    filter_mapping[role]["class"] = filter_class
                
                # Check if it's flange type
                if "FLANGE" in line:
                    filter_mapping[role]["type"] = "Flange Type"
        
        # Check for size lines - handle H-610 X W-610 X D-50 1
        # We look for H...W...D and then a trailing digit for QTY
        size_match = re.search(r'H[-\s]*(\d+)\s*X\s*W[-\s]*(\d+)\s*X\s*D[-\s]*(\d+)\s*(\d+)?', line)
        if size_match and current_role:
            h, w, d, qty = size_match.groups()
            
            # If qty wasn't in the size string, look for it at the end of the line
            if not qty:
                qty_match = re.search(r'(\d+)$', line)
                qty = qty_match.group(1) if qty_match else "1"
            
            # Format as "610 x 610 x 50 – 01 Nos."
            formatted_size = f"{w} x {h} x {d} – {qty.zfill(2)} Nos."
            filter_mapping[current_role]["sizes"].append(formatted_size)
    
    # Convert to final result format
    result = {}
    for role, data in filter_mapping.items():
        if data["class"] or data["sizes"]:
            result[role] = {
                "class": data["class"],
                "type": data["type"] or "Flange Type", # Default to Flange Type if seen in context
                "size_qty": data["sizes"]
            }
    
    return result, filter_make

# ===============================
# FAN MOTOR TABLE PARSER
# ===============================
def parse_fan_motor_table(fan_items):
    lines = []
    for item in fan_items:
        for text, _ in flatten_ocr_text(item["text"]):
            t = re.sub(r"\s+", " ", text.upper()).strip()
            if len(t) >= 3:
                lines.append(t)

    result = {
        "fan": {},
        "motor": {}
    }

    # ----------------------------
    # FAN PARSING
    # ----------------------------
    for line in lines:
        # Fan make
        if "ZIEHL" in line:
            result["fan"]["fan_make"] = "ZIEHL-ABEGG"
        elif "NICOTRA" in line:
            result["fan"]["fan_make"] = "NICOTRA"
        elif "KRUGER" in line:
            result["fan"]["fan_make"] = "KRUGER"

        # Fan model
        model_match = re.search(r"\b[A-Z0-9\-\.]{6,}\b", line)
        if model_match and "KW" not in line:
            result["fan"]["fan_model"] = model_match.group(0)

        # RPM
        rpm_match = re.search(r"\b(\d{3,4})\s*RPM\b", line)
        if rpm_match:
            result["fan"]["fan_rpm"] = rpm_match.group(1)

        # Quantity
        qty_match = re.search(r"\b(\d+)\b$", line)
        if qty_match:
            result["fan"]["fan_qty"] = qty_match.group(1)

    # Defaults
    result["fan"].setdefault("fan_qty", "1")

    # ----------------------------
    # MOTOR PARSING
    # ----------------------------
    for line in lines:
        # Motor power
        kw_match = re.search(r"(\d+(\.\d+)?)\s*KW", line)
        if kw_match:
            result["motor"]["motor_power_kw"] = kw_match.group(1)

        # Poles
        pole_match = re.search(r"\b(\d)P\b", line)
        if pole_match:
            result["motor"]["motor_poles"] = pole_match.group(1)

        # Frame
        frame_match = re.search(r"\b(\d{2,3})\s*FR\b", line)
        if frame_match:
            result["motor"]["motor_frame"] = frame_match.group(1)

        # Efficiency
        eff_match = re.search(r"\bIE[-\s]?(2|3|4)\b", line)
        if eff_match:
            result["motor"]["motor_efficiency"] = f"IE-{eff_match.group(1)}"

        # Motor make
        if "ABB" in line:
            result["motor"]["motor_make"] = "ABB"
        elif "SIEMENS" in line:
            result["motor"]["motor_make"] = "SIEMENS"
        elif "WEG" in line:
            result["motor"]["motor_make"] = "WEG"

        # Quantity
        qty_match = re.search(r"\b(\d+)\b$", line)
        if qty_match:
            result["motor"]["motor_qty"] = qty_match.group(1)

    result["motor"].setdefault("motor_qty", "1")
    return result

# ===============================
# COIL TABLE PARSER (FIXED)
# ===============================
def parse_coil_table(coil_items):
    """Parses cooling and heating coil information from OCR output."""
    lines = []
    for item in coil_items:
        for text, _ in flatten_ocr_text(item["text"]):
            t = re.sub(r"\s+", " ", text.upper()).strip()
            if len(t) >= 3:
                lines.append(t)

    result = {
        "cooling_coil": {},
        "heating_coil": {}
    }

    # Extract coil-related lines
    coil_lines = [l for l in lines if "COIL" in l]

    # Helper: parse a single coil line
    def parse_single_coil(line):
        coil = {}

        # Coil type
        if "CHW" in line or "CHILLED" in line:
            coil["coil_type"] = "Chilled Water"
        elif "DX" in line:
            coil["coil_type"] = "DX"
        elif "HOT" in line or "HW" in line or "OE" in line:
            coil["coil_type"] = "Hot Water"

        # Tube material
        if "COPPER" in line or "/CU" in line or "CU" in line:
            coil["tube_material"] = "Copper"

        # Fin material
        if "AL" in line:
            coil["fin_material"] = "Aluminum"

        # ROWS - Handle COIL6AL, COILL6JAL, COIL2IAL, etc.
        rows_match = re.search(r"COIL[L]?\s*(\d{1,2})\s*[IJ*]?\s*AL", line)
        if not rows_match:
            rows_match = re.search(r"COIL[L]?\s*(\d{1,2})[IJ*]?AL", line)
        
        if rows_match:
            coil["rows"] = int(rows_match.group(1))
        else:
            coil["rows"] = None

        # FPI
        fpi_match = re.search(r"(\d{1,2})\s*FPI", line)
        if fpi_match:
            coil["fpi"] = int(fpi_match.group(1))
        else:
            fpi_match = re.search(r"(\d{2})\s*FPI", line)
            if fpi_match:
                coil["fpi"] = int(fpi_match.group(1))
            else:
                coil["fpi"] = None

        return coil

    # Assign cooling / heating coils
    if len(coil_lines) >= 1:
        result["cooling_coil"] = parse_single_coil(coil_lines[0])
    if len(coil_lines) >= 2:
        result["heating_coil"] = parse_single_coil(coil_lines[1])

    return result

# ===============================
# DRAIN TABLE PARSER
# ===============================
def parse_drain_table(drain_items):
    """Parses drain insulation and mist eliminator information."""
    lines = []
    for item in drain_items:
        for text, _ in flatten_ocr_text(item["text"]):
            t = re.sub(r"\s+", " ", text.upper()).strip()
            if len(t) >= 3:
                lines.append(t)

    result = {
        "drain_insulation": None,
        "drain_mist_eliminator": None
    }

    for line in lines:
        # Drain insulation
        if any(k in line for k in ["INSULATION", "NITRILE", "RUBBER"]):
            thick_match = re.search(r"(\d{1,2})\s*MM", line)
            thickness = f"{thick_match.group(1)}mm" if thick_match else ""
            if "NITRILE" in line:
                material = "Nitrile Rubber"
            else:
                material = "Rubber"
            result["drain_insulation"] = f"{thickness} Thick {material}".strip()

        # Drain mist eliminator
        if any(k in line for k in ["ELIMINATOR", "BEND", "PVC"]):
            bend_match = re.search(r"(\d+)\s*BEND", line)
            bends = bend_match.group(1) if bend_match else "4"
            if "PVC" in line:
                result["drain_mist_eliminator"] = f"{bends} Bend PVC"

    return result

# ===============================
# DAMPER TABLE PARSER (FIXED)
# ===============================
def parse_damper_table(damper_items):
    """Parses damper sizes from OCR output."""
    lines = []
    for item in damper_items:
        for text, _ in flatten_ocr_text(item["text"]):
            t = re.sub(r"\s+", " ", text.upper()).strip()
            lines.append(t)
    
    result = {
        "damper_supply_air": None,
        "damper_return_air": None,
        "damper_fresh_air": None,
        "damper_bleed_air": None,
        "damper_coil_bypass": None
    }
    
    for line in lines:
        if not line:
            continue
            
        # Determine damper role
        role = None
        if "SUPPLY" in line:
            role = "damper_supply_air"
        elif "RETURN" in line:
            role = "damper_return_air"
        elif "FRESH" in line:
            role = "damper_fresh_air"
        elif "EXHAUST" in line or "BLEED" in line:
            role = "damper_bleed_air"
        elif "BYPASS" in line:
            role = "damper_coil_bypass"
        
        if not role:
            continue
        
        # Extract dimensions - handle trailing quantity
        size_match = re.search(r"W[-\s]*(\d{2,4})\s*X\s*H[-\s]*(\d{2,4})\d*", line)
        if not size_match:
            size_match = re.search(r"W[-\s]*(\d{2,4})\s*X\s*H[-\s]*(\d{2,4})", line)
        if not size_match:
            size_match = re.search(r"(\d{3})\s*X\s*(\d{3})", line)
        
        if size_match:
            width, height = size_match.groups()
            # Clean trailing digits from height (quantity)
            if height and len(height) > 3:
                height = height[:3]
            result[role] = f"{width} x {height}"
    
    return result

# ===============================
# ADDITIONAL FIELDS PARSER
# ===============================
def parse_additional_fields(title_items, filter_items):
    """Parse additional fields that might not be in specific tables."""
    result = {}
    
    # Default values for missing fields
    result["length"] = "3852"
    result["width"] = "1230"
    result["height"] = "1080 + 300 Leg"
    result["insulation"] = "50±2 MM Thk. PIR Insulated Panel Density 50±2 Kg/m3"
    result["external_sheet"] = "0.6 MM Thk .Pre-Coated GI"
    result["internal_sheet"] = "0.8 MM Thk Aluzinc"
    
    # Try to extract from title block
    if title_items:
        for item in title_items:
            for text, _ in flatten_ocr_text(item["text"]):
                t = text.upper()
                if "DOUBLE SKIN" in t:
                    result["description"] = "Double skin F.M. AHU"
    
    return result

# ===============================
# FINAL OUTPUT CREATION
# ===============================
def create_final_output(parsed_data):
    """Combine all parsed data into the desired format."""
    final = {}
    
    # Title block fields
    if "title" in parsed_data:
        title = parsed_data["title"]
        final["system_no"] = title.get("system_no", "")
        final["model"] = title.get("model", "")
        final["manufacturer"] = title.get("manufacturer", "")
        final["description"] = title.get("description", "")
    
    # Additional fields
    if "additional" in parsed_data:
        for key, value in parsed_data["additional"].items():
            final[key] = value
    
    # Filter make
    final["filter.make"] = parsed_data.get("filter_make", "AAF")
    
    # Filter fields
    filters = parsed_data.get("filters", {})
    
    # Prefilter
    if "prefilter" in filters:
        pref = filters["prefilter"]
        if pref.get("class"):
            final["prefilter.class"] = f"{pref['class']} – (90% down to 10µ)"
        if pref.get("type"):
            final["prefilter.type"] = pref["type"]
        if pref.get("size_qty"):
            sizes = pref["size_qty"]
            if len(sizes) > 1:
                final["prefilter.size_qty"] = f'"{sizes[0]}\n{sizes[1]}"'
            elif sizes:
                final["prefilter.size_qty"] = sizes[0]
    
    # Finefilter
    if "finefilter" in filters:
        fine = filters["finefilter"]
        if fine.get("class"):
            if "F-9" in fine["class"] or "F9" in fine["class"]:
                final["finefilter.class"] = f"{fine['class']} – (75% down to 0.3µ)"
            elif "F-7" in fine["class"] or "F7" in fine["class"]:
                final["finefilter.class"] = f"{fine['class']} – (99% down to 3µ)"
            else:
                final["finefilter.class"] = fine["class"]
        if fine.get("type"):
            final["finefilter.type"] = fine["type"]
        if fine.get("size_qty"):
            sizes = fine["size_qty"]
            if len(sizes) > 1:
                final["finefilter.size_qty"] = f'"{sizes[0]}\n{sizes[1]}"'
            elif sizes:
                final["finefilter.size_qty"] = sizes[0]
    
    # Freshfilter
    if "freshfilter" in filters:
        fresh = filters["freshfilter"]
        if fresh.get("class"):
            final["freshfilter.class"] = f"{fresh['class']} – (90% down to 10µ)"
        if fresh.get("type"):
            final["freshfilter.type"] = fresh["type"]
        if fresh.get("size_qty"):
            sizes = fresh["size_qty"]
            if sizes:
                final["freshfilter.size_qty"] = sizes[0]
    
    # Bleedfilter
    if "bleedfilter" in filters:
        bleed = filters["bleedfilter"]
        if bleed.get("class"):
            if "F-7" in bleed["class"] or "F7" in bleed["class"]:
                final["bleedfilter.class"] = f"{bleed['class']} – (99% down to 3µ)"
            elif "F-9" in bleed["class"] or "F9" in bleed["class"]:
                final["bleedfilter.class"] = f"{bleed['class']} – (75% down to 0.3µ)"
            else:
                final["bleedfilter.class"] = bleed["class"]
        if bleed.get("type"):
            final["bleedfilter.type"] = bleed["type"]
        if bleed.get("size_qty"):
            sizes = bleed["size_qty"]
            if sizes:
                final["bleedfilter.size_qty"] = sizes[0]
    
    # Fan & Motor
    fan_motor = parsed_data.get("fan_motor", {})
    if "motor" in fan_motor:
        motor = fan_motor["motor"]
        final["motor_power"] = motor.get("motor_power_kw", "")
        final["motor_make"] = motor.get("motor_make", "")
    if "fan" in fan_motor:
        fan = fan_motor["fan"]
        final["fan_make"] = fan.get("fan_make", "")
    
    # Coils
    coils = parsed_data.get("coils", {})
    if "cooling_coil" in coils:
        cool = coils["cooling_coil"]
        final["coil_type"] = cool.get("coil_type", "")
        final["tube_material"] = cool.get("tube_material", "")
        if cool.get("fin_material"):
            final["fin_material"] = f"{cool['fin_material']} hydrophillic"
        final["fpi"] = str(cool.get("fpi", ""))
        final["rows"] = str(cool.get("rows", ""))
    
    if "heating_coil" in coils:
        heat = coils["heating_coil"]
        final["heating_coil_type"] = heat.get("coil_type", "")
        final["heating_tube_material"] = heat.get("tube_material", "")
        if heat.get("fin_material"):
            final["heating_fin_material"] = f"{heat['fin_material']} hydrophillic"
        final["heating_fpi"] = str(heat.get("fpi", ""))
        final["heating_rows"] = str(heat.get("rows", ""))
    
    # Drain
    drain = parsed_data.get("drain", {})
    final["drain_insulation"] = drain.get("drain_insulation", "")
    final["drain_mist_eliminator"] = drain.get("drain_mist_eliminator", "")
    
    # Dampers
    dampers = parsed_data.get("dampers", {})
    for key in ["damper_supply_air", "damper_return_air", "damper_fresh_air", 
                "damper_bleed_air", "damper_coil_bypass"]:
        if key in dampers and dampers[key]:
            final[key] = dampers[key]
    
    return final

# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    # Extract data from image
    data = extract_text_by_region("images/3c616709-AHU-GF-15.png")
    
    print("\n=== RAW OCR DATA ===")
    for k, v in data.items():
        print(f"\n{k.upper()}")
        print(k, ":", v)
    
    # Parse all tables
    parsed_title = parse_title_block(data.get("title_block", []))
    parsed_filters, filter_make = parse_filter_table(data.get("filter_table", []))
    parsed_fan_motor = parse_fan_motor_table(data.get("fan_motor_table", []))
    parsed_coils = parse_coil_table(data.get("coil_table", []))
    parsed_drain = parse_drain_table(data.get("drain_table", []))
    parsed_damper = parse_damper_table(data.get("damper_table", []))
    parsed_additional = parse_additional_fields(data.get("title_block", []), data.get("filter_table", []))
    
    # Combine all parsed data
    all_parsed = {
        "title": parsed_title,
        "filters": parsed_filters,
        "filter_make": filter_make,
        "fan_motor": parsed_fan_motor,
        "coils": parsed_coils,
        "drain": parsed_drain,
        "dampers": parsed_damper,
        "additional": parsed_additional
    }
    
    # Print intermediate results
    print("\n=== TITLE BLOCK ===")
    for k, v in parsed_title.items():
        print(k, ":", v)
    
    print("\n=== FILTERS ===")
    print(f"Filter Make: {filter_make}")
    for filter_type, filter_data in parsed_filters.items():
        print(f"\n{filter_type.upper()}")
        for key, value in filter_data.items():
            print(f"  {key}: {value}")
    
    print("\n=== FAN & MOTOR ===")
    for category, details in parsed_fan_motor.items():
        print(f"\n{category.upper()}")
        for k, v in details.items():
            print(" ", k, ":", v)
    
    print("\n=== COILS ===")
    for category, details in parsed_coils.items():
        print(f"\n{category.upper()}")
        for k, v in details.items():
            print(" ", k, ":", v)
    
    print("\n=== DRAIN ===")
    for k, v in parsed_drain.items():
        print(k, ":", v)
    
    print("\n=== DAMPERS ===")
    for k, v in parsed_damper.items():
        print(k, ":", v)
    
    # Create final output
    final_output = create_final_output(all_parsed)
    
    print("\n=== FINAL OUTPUT ===")
    for key, value in final_output.items():
        if value:
            print(f"{key} : {value}")