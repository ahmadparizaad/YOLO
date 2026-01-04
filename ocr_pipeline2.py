import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import re
from collections import Counter

# Global reader instance (lazy-loaded)
_easyocr_reader = None

def get_easyocr_reader():
    """Lazy-loads the EasyOCR reader to save startup time."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        # We'll use this as a fallback for dimension extraction
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader

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

def ocr_image(img, psm=6):
    if img is None or img.size == 0:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    config = (
        f"--psm {psm} "
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

            if label in ["filter_table", "coil_table", "damper_table"]:
                # Use grid extraction for structured tables to get individual cells
                all_rows_text = extract_table_grid(cropped)
            else:
                # Use standard row extraction for others
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
# TABLE GRID EXTRACTION (NEW)
# ===============================
def extract_table_grid(img):
    """
    Detects horizontal and vertical lines to extract individual cells.
    Returns a list of rows, where each row is a list of cell texts:
    [ [('Type', 1.0), ('Rating', 1.0), ...], ... ]
    """
    # 1. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to get binary image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)

    # 2. Detect Horizontal Lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1]//20, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # 3. Detect Vertical Lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[0]//20))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # 4. Combine to form Grid
    grid = cv2.add(h_lines, v_lines)
    
    # Dilate slightly to close gaps in lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    grid = cv2.dilate(grid, kernel, iterations=1)

    # 5. Find Contours (Cells)
    # Invert grid so lines are black and cells are white blobs
    grid_inv = cv2.bitwise_not(grid)
    contours, _ = cv2.findContours(grid_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to remove noise (too small)
    cells = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out small noise and the main bounding box itself
        if w > 20 and h > 10 and w < img.shape[1] * 0.9:
            cells.append((x, y, w, h))

    # 6. Sort Cells into Rows and Columns
    # Sort by Y first
    cells.sort(key=lambda k: k[1])

    rows = []
    current_row = []
    last_y = -100

    for x, y, w, h in cells:
        # If this cell is significantly lower than the last one, start a new row
        if y > last_y + (h / 2): 
            if current_row:
                # Sort the previous row by X (columns)
                current_row.sort(key=lambda k: k[0])
                rows.append(current_row)
            current_row = []
            last_y = y
        current_row.append((x, y, w, h))

    # Append the last row
    if current_row:
        current_row.sort(key=lambda k: k[0])
        rows.append(current_row)

    # 7. OCR Each Cell
    grid_data = []
    for row_cells in rows:
        row_text_list = []
        for i, (x, y, w, h) in enumerate(row_cells):
            # Crop cell
            cell_img = img[y:y+h, x:x+w]
            
            # Check if it's the last column (Quantity) or a NARROW cell (likely single digit like RD/Rows)
            is_last_col = (i == len(row_cells) - 1)
            
            # Detect narrow cells: width < 50px OR aspect ratio (h/w) > 1.5
            # These typically contain single digits (e.g., Rows column "RD" with values 2, 4, 6, 8)
            is_narrow_cell = (w < 50) or (h > 0 and w > 0 and (h / w) > 1.5)
            
            if is_last_col or is_narrow_cell:
                # === SPECIAL HANDLING FOR SMALL DIGIT CELLS (Quantity, Rows, etc.) ===
                # 1. "Macro Zoom": Resize to fixed height of 60px to make digits clear
                target_height = 60
                scale = target_height / float(h) if h > 0 else 1
                new_w = max(int(w * scale), 30)  # Ensure minimum width
                cell_img = cv2.resize(cell_img, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)
                
                # 2. Add generous padding (Tesseract needs whitespace around digits)
                cell_img = cv2.copyMakeBorder(cell_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                
                # 3. OCR with Digit-Optimized Settings
                # PSM 7 (Single Line) is robust for "1" or "2"
                # PSM 10 (Single Character) as fallback for very small cells
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                # Simple thresholding often works better for digits than adaptive
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Try PSM 7 first (single line), then PSM 10 (single char) if no result
                text = ""
                for psm in [7, 10, 13]:
                    config = f"--psm {psm} -c tessedit_char_whitelist=0123456789Nos. "
                    text = pytesseract.image_to_string(binary, config=config).strip()
                    # Clean the result - keep only digits
                    digits_only = re.sub(r"[^\d]", "", text)
                    if digits_only:
                        text = digits_only
                        break
                
                if text:
                    row_text_list.append((text, 1.0))
                else:
                    row_text_list.append(("", 0.0))
            else:
                # === STANDARD HANDLING FOR OTHER COLUMNS ===
                # Add white border for better OCR
                cell_img = cv2.copyMakeBorder(cell_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                
                # OCR the cell
                # Use PSM 6 (Block) by default to handle multi-line text
                cell_ocr = ocr_image(cell_img, psm=6)
                
                full_cell_text = " ".join([t[0] for t in cell_ocr])
                
                if full_cell_text.strip():
                    row_text_list.append((full_cell_text, 1.0))
                else:
                    row_text_list.append(("", 0.0)) # Empty cell
        
        if any(t[0] for t in row_text_list): # Only add non-empty rows
            grid_data.append(row_text_list)

    return grid_data

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

    # System No - Improved extraction
    # Strategy 1: Look for "AHU REF" label followed by actual system number
    # Handle both spaced "AHU REF. AHU-GF-06" and concatenated "AHUREF.AHU-GF-06"
    for line in lines:
        # Match patterns like "AHUREF. AHU-GF-06" or "AHU REF.: AHU-GF-06" 
        # Also handle no space: "AHUREF.AHU-GF-06"
        ref_match = re.search(r"AHU\s*REF\.?\s*:?\s*(AHU-[A-Z]{1,3}-\d{1,2})", line, re.I)
        if ref_match:
            result["system_no"] = ref_match.group(1).upper()
            break
        # Handle concatenated case without proper separator: "AHUREF.AHU-GF-06"
        ref_match = re.search(r"AHUREF\.?(AHU-[A-Z]{1,3}-\d{1,2})", line.replace(" ", ""), re.I)
        if ref_match:
            result["system_no"] = ref_match.group(1).upper()
            break
    
    # Strategy 2: Look for specific AHU-XX-## pattern in lines (more precise than full_text)
    if "system_no" not in result:
        for line in lines:
            m = re.search(r"\b(AHU-[A-Z]{1,3}-\d{1,2})\b", line)
            if m:
                result["system_no"] = m.group(1)
                break
    
    # Strategy 3: Look in full text as fallback
    if "system_no" not in result:
        m = re.search(r"\bAHU-[A-Z]{1,3}-\d{1,2}\b", full_text)
        if m:
            result["system_no"] = m.group(0)
    
    # Strategy 3: Fallback - look in each line for the pattern after common labels
    if "system_no" not in result:
        for line in lines:
            # Skip lines that are just labels
            if re.match(r"^AHU\s*REF\.?:?$", line, re.I):
                continue
            # Look for standalone system number pattern
            m = re.search(r"\b(AHU-[A-Z0-9]+-\d+)\b", line)
            if m and "REF" not in m.group(1).upper():
                result["system_no"] = m.group(1)
                break

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
    if "SEMI HEPA" in desc or "SEMIHEPA" in desc:
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
        # Handle new structured grid format (list of lists of tuples)
        if item["text"] and isinstance(item["text"][0], list) and isinstance(item["text"][0][0], tuple):
            for row in item["text"]:
                # Join cells in the row with spaces to ensure separation
                row_text = " ".join([t[0] for t in row])
                t = re.sub(r"\s+", " ", row_text.upper()).strip()
                if t:
                    lines.append(t)
        else:
            # Fallback for old format
            for text, _ in flatten_ocr_text(item["text"]):
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
        "prefilter": {"class": None, "type": None, "sizes": [], "rating": None},
        "freshfilter": {"class": None, "type": None, "sizes": [], "rating": None},
        "finefilter": {"class": None, "type": None, "sizes": [], "rating": None},
        "bleedfilter": {"class": None, "type": None, "sizes": [], "rating": None}
    }
    
    # Current filter being processed
    current_role = None
    
    for line in lines:
        # Skip header lines
        if "FILTER DETAILS" in line or "RATING" in line or "QTY." in line:
            continue
        
        # Fix common OCR error: 305 -> 3505
        line = line.replace("3505", "305")
        
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
                
                # Extract Rating
                # Look for pattern like 90% to 10mic or 90%to10mic
                rating_match = re.search(r"(\d{1,4}(?:\.\d+)?)%.*?MIC\.?", line)
                if rating_match:
                    raw_rating = rating_match.group(0)
                    
                    # Fix percentage if needed (e.g. 715% -> 75%)
                    pct_match = re.match(r"^(\d+)", raw_rating)
                    if pct_match:
                        pct_str = pct_match.group(1)
                        pct_val = int(pct_str)
                        
                        if pct_val > 100:
                            # Heuristic: 715 -> 75, 910 -> 90 (remove middle '1')
                            s_val = str(pct_val)
                            new_val = None
                            
                            if len(s_val) == 3 and s_val[1] == '1':
                                new_val = s_val[0] + s_val[2]
                            
                            if new_val and 50 <= int(new_val) <= 100:
                                raw_rating = raw_rating.replace(pct_str + "%", new_val + "%")

                    # Fix "TOO" -> "TO 0" (common OCR error for "TO 0")
                    raw_rating = raw_rating.replace("TOO", "TO 0")

                    filter_mapping[role]["rating"] = raw_rating

                # Check if it's flange type
                if "FLANGE" in line:
                    filter_mapping[role]["type"] = "Flange Type"
        
        # Check for size lines - handle H-610 X W-610 X D-50 1
        # We look for H...W...D and then a trailing digit for QTY
        # We remove the optional 4th group from regex to avoid capturing part of D as qty
        size_match = re.search(r'H[-\s]*(\d+)\s*X\s*W[-\s]*(\d+)\s*X\s*D[-\s]*(\d+)', line)
        if size_match and current_role:
            h, w, d = size_match.groups()
            
            # Look for quantity in the text *after* the size match
            remaining_text = line[size_match.end():].strip()
            qty_match = re.search(r'(\d+)', remaining_text)
            
            if qty_match:
                qty = qty_match.group(1)
            else:
                qty = "1"
            
            # Format as "610 x 610 x 50 – 01 Nos."
            formatted_size = f"{h} x {w} x {d} – {qty.zfill(2)} Nos."
            filter_mapping[current_role]["sizes"].append(formatted_size)
    
    # Convert to final result format
    result = {}
    for role, data in filter_mapping.items():
        if data["class"] or data["sizes"]:
            result[role] = {
                "class": data["class"],
                "type": data["type"] or "Flange Type", # Default to Flange Type if seen in context
                "size_qty": data["sizes"],
                "rating": data["rating"]
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
    """Parses cooling and heating coil information from OCR output.
    
    Expected table structure (columns):
    TYPE | RD (Rows) | FIN HYDROPHILIC | HEADER MAT | CASING MAT | SIZE
    
    Row A: CHW Coil (Cooling) - typically 6 rows
    Row B: HW Coil (Heating) - typically 2 rows  
    Row C: Eliminator
    
    Since grid extraction may not cleanly separate cells, we combine all cells
    into a row string and use smart regex patterns.
    """
    result = {
        "cooling_coil": {},
        "heating_coil": {}
    }
    
    def parse_coil_row(row_text, cells):
        """Parse a coil row intelligently from combined text and individual cells."""
        coil = {}
        row_text = row_text.upper()
        
        # Determine coil type
        if "CHW" in row_text or "CHILLED" in row_text:
            coil["coil_type"] = "Chilled Water"
            coil["_target"] = "cooling_coil"
        elif "HW" in row_text or "HOT" in row_text:
            coil["coil_type"] = "Hot Water"
            coil["_target"] = "heating_coil"
        elif "DX" in row_text:
            coil["coil_type"] = "DX"
            coil["_target"] = "cooling_coil"
        else:
            return None  # Not a coil row
        
        # Extract ROWS - multiple strategies
        # Typical values: 2, 4, 6, 8 for heating/cooling coils
        
        # Strategy 1: Look in individual cells for pure digit cells (from narrow cell extraction)
        # The grid extraction now applies macro-zoom to narrow cells, giving cleaner digits
        for idx, cell in enumerate(cells):
            cell_clean = cell.strip()
            # Pure single digit (the macro-zoom technique should give us clean "2", "6", etc.)
            if cell_clean.isdigit() and len(cell_clean) == 1 and cell_clean in "123456789":
                coil["rows"] = int(cell_clean)
                break
            # Also handle digits with OCR artifacts like "6)", "2.", "(6"
            digits_only = re.sub(r"[^\d]", "", cell_clean)
            if len(digits_only) == 1 and digits_only in "123456789":
                # Make sure this isn't part of a larger number context (like FPI)
                if "FPI" not in cell_clean.upper() and len(cell_clean) <= 3:
                    coil["rows"] = int(digits_only)
                    break
        
        # Strategy 2: "COIL" followed by digit (e.g., "CHWCOIL6")
        if "rows" not in coil:
            rows_match = re.search(r"COIL[L]?\s*(\d)[^\d]", row_text)
            if rows_match:
                coil["rows"] = int(rows_match.group(1))
        
        # Strategy 3: Digit immediately before "AL" (e.g., "6AL" or "2 AL")  
        if "rows" not in coil:
            rows_match = re.search(r"(\d)\s*AL(?!\.?\d)", row_text)  # Avoid matching "12FPI"
            if rows_match:
                digit = int(rows_match.group(1))
                if 1 <= digit <= 9:  # Typical coil row counts
                    coil["rows"] = digit
        
        # Strategy 4: Look for isolated digit in row text (not part of FPI)
        if "rows" not in coil:
            # Remove FPI patterns first to avoid false matches
            text_no_fpi = re.sub(r"\d{1,2}\s*FPI", "", row_text)
            rows_match = re.search(r"\b([1-8])\b", text_no_fpi)
            if rows_match:
                coil["rows"] = int(rows_match.group(1))
        
        # Extract FPI
        fpi_match = re.search(r"(\d{1,2})\s*FPI", row_text)
        if fpi_match:
            coil["fpi"] = int(fpi_match.group(1))
        
        # Fin material (if AL mentioned with FPI context)
        if "AL" in row_text:
            coil["fin_material"] = "Aluminum"
        
        # Tube/Header material
        if "COPPER" in row_text:
            coil["tube_material"] = "Copper"
        
        return coil
    
    for item in coil_items:
        text_data = item.get("text", [])
        
        if not text_data or not isinstance(text_data, list):
            continue
        
        # Process each row
        for row in text_data:
            cells = []
            row_text = ""
            
            # Handle grid format: [[('cell1', conf), ('cell2', conf)], ...]
            if isinstance(row, list) and len(row) > 0:
                if isinstance(row[0], tuple):
                    cells = [cell[0].upper().strip() for cell in row]
                    row_text = " ".join(cells)
                else:
                    # Nested list case
                    continue
            # Handle flat format: [('text', conf), ...]
            elif isinstance(row, tuple) and len(row) == 2:
                row_text = row[0].upper().strip()
                cells = [row_text]
            else:
                continue
            
            # Skip header rows
            if "TYPE" in row_text or "COIL DETAILS" in row_text or "HEADER" in row_text:
                continue
            
            # Parse the row
            coil = parse_coil_row(row_text, cells)
            if coil:
                target_key = coil.pop("_target", None)
                if target_key:
                    # Merge into result (don't overwrite existing data)
                    for k, v in coil.items():
                        if v is not None and k not in result[target_key]:
                            result[target_key][k] = v
    
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
    """Parses damper sizes from OCR output using grid extraction.
    
    Expected table structure (columns):
    TYPE | MATERIAL | SIZE | QTY
    
    Damper types:
    - SUPPLY AIR DAMPER
    - RETURN AIR DAMPER  
    - FRESH AIR DAMPER
    - EXHAUST AIR DAMPER (bleed)
    - BYPASS AIR DAMPER (coil bypass)
    """
    result = {
        "damper_supply_air": None,
        "damper_return_air": None,
        "damper_fresh_air": None,
        "damper_bleed_air": None,
        "damper_coil_bypass": None
    }
    
    def determine_damper_role(text):
        """Determine damper role from text, handling OCR variations."""
        text = text.upper()
        # Handle common OCR errors and variations
        # BYPASS variations: RYPASS, BYEPASS, BY PASS, BYPAS, etc.
        text = text.replace("RYPASS", "BYPASS")  # R misread as B
        text = text.replace("BYEPASS", "BYPASS")
        text = text.replace("BY PASS", "BYPASS")
        text = text.replace("BYPAS ", "BYPASS ")
        # SUPPLY variations
        text = text.replace("SUPPIY", "SUPPLY").replace("SUPFLY", "SUPPLY")
        text = text.replace("SUPPLT", "SUPPLY").replace("SIJPPLY", "SUPPLY")
        # RETURN variations  
        text = text.replace("RETUEN", "RETURN").replace("RETURA", "RETURN")
        text = text.replace("RETIJRN", "RETURN")
        # DAMPER variations: NAMPFR, DAMPFR, DAMFER, etc.
        text = text.replace("NAMPFR", "DAMPER").replace("DAMPFR", "DAMPER")
        text = text.replace("DAMFER", "DAMPER").replace("DAMPEF", "DAMPER")
        
        if "SUPPLY" in text:
            return "damper_supply_air"
        elif "RETURN" in text:
            return "damper_return_air"
        elif "FRESH" in text:
            return "damper_fresh_air"
        elif "EXHAUST" in text or "BLEED" in text:
            return "damper_bleed_air"
        elif "BYPASS" in text or "BYPA" in text or "YPASS" in text:
            return "damper_coil_bypass"
        return None
    
    def extract_dimensions(text):
        """Extract W x H dimensions from text, handling OCR errors."""
        original_text = text.upper()
        text = original_text.replace(" ", "")
        
        # Pattern 1: W-2000XH-500 (with dashes and W/H labels) - try clean match first
        match = re.search(r"W-?(\d{2,4})X+H-?(\d{2,4})", text)
        if match:
            w, h = match.group(1), match.group(2)
            return f"{w} x {h}"
        
        # Pattern 2: Handle OCR errors like "5S00" -> "500" 
        # The key insight: typical damper dimensions are 100-2500mm
        # So "5S00" (5500) is likely "500", and "2S00" is likely "200"
        def fix_ocr_digit(val):
            """Fix OCR errors in dimension values."""
            val = val.upper()
            # Replace S/$ with 5 when between digits (e.g., "5S0" -> "550")
            # But also check if result is unrealistic
            fixed = re.sub(r"[S$]", "5", val)
            fixed = re.sub(r"[O]", "0", fixed)
            
            # If all digits now, check plausibility
            if fixed.isdigit():
                num = int(fixed)
                # If > 3000 and original had S, likely S was noise - try removing it
                if num > 3000 and 'S' in val:
                    # Try removing the S instead of replacing
                    no_s = re.sub(r"[S$]", "", val)
                    if no_s.isdigit() and 100 <= int(no_s) <= 2500:
                        return no_s
                return fixed
            return val
        
        # Look for dimension patterns with potential OCR errors
        # Pattern: W-2000XH-5S00 where S might be misread
        match = re.search(r"W-?([0-9S$O]{2,4})X+H-?([0-9S$O]{2,4})", text)
        if match:
            w_raw, h_raw = match.group(1), match.group(2)
            w = fix_ocr_digit(w_raw)
            h = fix_ocr_digit(h_raw)
            if w.isdigit() and h.isdigit():
                return f"{w} x {h}"
        
        # Pattern 3: Plain numbers with X separator
        match = re.search(r"(\d{3,4})X(\d{3,4})", text)
        if match:
            return f"{match.group(1)} x {match.group(2)}"
        
        # Pattern 4: Smaller dampers (2-4 digits x 2-4 digits)
        match = re.search(r"(\d{2,4})[xX](\d{2,4})", text)
        if match:
            return f"{match.group(1)} x {match.group(2)}"
        return None
    
    for item in damper_items:
        text_data = item.get("text", [])
        
        # Check if we have grid data (list of lists of tuples)
        if text_data and isinstance(text_data, list) and len(text_data) > 0:
            if isinstance(text_data[0], list) and len(text_data[0]) > 0 and isinstance(text_data[0][0], tuple):
                # Grid format: [[('cell1', conf), ('cell2', conf)], ...]
                for row in text_data:
                    if len(row) < 2:
                        continue
                    
                    # Combine all cells in row to find role and dimensions
                    row_text = " ".join([cell[0] for cell in row])
                    cells = [cell[0].upper().strip() for cell in row]
                    
                    # Determine role from first cell or combined text
                    role = determine_damper_role(cells[0]) or determine_damper_role(row_text)
                    
                    if not role:
                        continue
                    
                    # Look for dimensions in all cells
                    dims = None
                    for cell in cells:
                        dims = extract_dimensions(cell)
                        if dims:
                            break
                    
                    # Also try combined row text
                    if not dims:
                        dims = extract_dimensions(row_text)
                    
                    if dims and not result[role]:
                        result[role] = dims
            else:
                # Fallback: old line-based format
                for text, _ in flatten_ocr_text(text_data):
                    line = re.sub(r"\s+", " ", text.upper()).strip()
                    if not line:
                        continue
                    
                    role = determine_damper_role(line)
                    if role:
                        dims = extract_dimensions(line)
                        if dims and not result[role]:
                            result[role] = dims
    
    return result

# ===============================
# ADDITIONAL FIELDS PARSER
# ===============================
def parse_additional_fields(title_items, filter_items, dimensions=None):
    """Parse additional fields that might not be in specific tables."""
    result = {}
    
    # Dimensions
    if dimensions:
        if dimensions.get("length"):
            result["length"] = str(dimensions["length"])
        if dimensions.get("width"):
            result["width"] = str(dimensions["width"])
        
        formatted_height = format_height_with_legs(dimensions.get("height"), dimensions.get("legs"))
        if formatted_height:
            result["height"] = formatted_height

    # Default values for other fields
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
# Helper: Remove blue hatching using color filtering
# ===============================
def remove_blue_hatching(img):
    """
    Filters out blue lines from CAD drawings using HSV color space.
    Keeps only black/dark text, removes colored hatching.
    """
    if img is None or img.size == 0:
        return img
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Blue in HSV: Hue ~100-130, Saturation > 50
    # Create mask to detect blue pixels
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Replace blue pixels with white
    result = img.copy()
    result[blue_mask > 0] = [255, 255, 255]
    
    return result

# ===============================
# Helper: Rotate image for vertical text
# ===============================
def rotate_for_vertical_text(img):
    """Rotate 90 degrees clockwise so vertical text becomes horizontal."""
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# ===============================
# Helper: extract numbers from ROI
# =============================
def extract_numbers_from_roi(img, rotate=False, remove_hatching=False, debug_name=None, use_easyocr=False):
    """
    Extract numbers from an ROI with optional preprocessing.
    
    Args:
        img: Input image (BGR)
        rotate: If True, rotate 90° CW for vertical text
        remove_hatching: If True, filter out blue hatching lines
        debug_name: If provided, save intermediate images for debugging
        use_easyocr: If True, use EasyOCR as a fallback/verification
    
    Returns:
        List of integers found in the image
    """
    if img is None or img.size == 0:
        return [], Counter()
    
    # Step 1: Remove blue hatching if needed
    if remove_hatching:
        img = remove_blue_hatching(img)
    
    # Step 2: Rotate for vertical text if needed
    if rotate:
        img = rotate_for_vertical_text(img)
    
    # Save debug image if requested
    if debug_name:
        cv2.imwrite(f"debug_{debug_name}_preprocessed.png", img)
    
    # Step 3: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 4: Upscale for better OCR (6x with Lanczos for sharper edges)
    scale = 6
    gray_scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    
    # Step 4.5: Sharpen the image to enhance digit edges
    # Using unsharp masking with stronger effect
    blurred = cv2.GaussianBlur(gray_scaled, (0, 0), 3)
    gray_scaled = cv2.addWeighted(gray_scaled, 2.0, blurred, -1.0, 0)
    
    # Try multiple preprocessing approaches and combine results
    all_nums = []
    
    # Approach 1: Simple Otsu threshold (no blur)
    _, binary1 = cv2.threshold(gray_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary1 = cv2.copyMakeBorder(binary1, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Save debug binary if requested (use Otsu version)
    if debug_name:
        cv2.imwrite(f"debug_{debug_name}_binary.png", binary1)

    # Try each binary image with multiple PSM modes
    # Track frequency of each number found
    num_counts = Counter()
    
    for binary, approach_name in [(binary1, "otsu")]:
        # PSM 11, 6, 3 identified as top performers in testing
        for psm in [11, 6, 3]:
            config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(binary, config=config)
            
            # Extract numbers (2-5 digits)
            nums = [int(x) for x in re.findall(r"\d{2,5}", text)]
            for n in nums:
                num_counts[n] += 1
            if debug_name and nums:
                print(f"DEBUG {debug_name} {approach_name} PSM{psm}: OCR='{text.strip()}' nums={nums}")
    
    # Get all unique numbers
    all_nums = list(num_counts.keys())
    
    if debug_name:
        print(f"DEBUG {debug_name}: combined nums={all_nums}")
        # Show counts for plausible dimension values
        plausible = {k: v for k, v in num_counts.items() if 1000 <= k <= 6000}
        if plausible:
            print(f"DEBUG {debug_name}: plausible counts={dict(sorted(plausible.items(), key=lambda x: -x[1]))}")
    
    return all_nums, num_counts, gray_scaled

# ===============================
# Core function: extract diagram dimensions
# =============================
def extract_dimensions(image, elevation_box, plan_box, debug=False):
    """
    elevation_box, plan_box = (x1, y1, x2, y2)
    returns: dict with length, width, height, legs
    """

    result = {
        "length": None,
        "width": None,
        "height": None,
        "legs": None
    }

    # ------------------------
    # Elevation view ROIs
    # Note: Dimension text is often OUTSIDE the YOLO box boundaries
    # We need to extend beyond the detected box to capture dimensions
    # ------------------------
    ex1, ey1, ex2, ey2 = map(int, elevation_box)
    ew, eh = ex2 - ex1, ey2 - ey1
    
    # Get image dimensions to prevent out-of-bounds
    img_h, img_w = image.shape[:2]

    # LENGTH - horizontal text at bottom of elevation view
    # Capture a wider horizontal band at the bottom
    length_roi = image[
        int(ey1 + 0.75*eh):min(int(ey2 + 50), img_h),  # Bottom 25% + some extra
        int(ex1 + 0.10*ew):int(ex1 + 0.90*ew)  # Wide horizontal band
    ]

    # HEIGHT - vertical text to the RIGHT of the elevation box
    # Capture a tall strip on the right side
    height_roi = image[
        int(ey1):int(ey1 + 0.65*eh),  # Top 65% of elevation
        max(int(ex2 - 200), 0):min(int(ex2 + 400), img_w)  # Right edge with extension
    ]

    # LEGS - vertical text at bottom right of elevation
    # Dimensions: 300 (legs), 100 (base frame) - located far right of elevation
    legs_y1 = int(ey1 + 0.65*eh)
    legs_y2 = min(int(ey2 + 80), img_h)  # Extended lower
    legs_x1 = max(int(ex2 - 50), 0)      # Start just inside the box
    legs_x2 = min(int(ex2 + 400), img_w)  # Extend well past the box edge
    
    legs_roi = image[legs_y1:legs_y2, legs_x1:legs_x2]

    # ------------------------
    # Plan view ROI
    # ------------------------
    px1, py1, px2, py2 = map(int, plan_box)
    pw, ph = px2 - px1, py2 - py1

    # WIDTH - vertical text to the RIGHT of the plan view box
    width_roi = image[
        int(py1 + 0.10*ph):int(py1 + 0.90*ph),  # Most of the height
        max(int(px2 - 200), 0):min(int(px2 + 400), img_w)  # Right edge with extension
    ]

    # Save raw ROIs for debugging (only if debug enabled)
    if debug:
        cv2.imwrite("debug_length_raw.png", length_roi)
        cv2.imwrite("debug_height_raw.png", height_roi)
        cv2.imwrite("debug_legs_raw.png", legs_roi)
        cv2.imwrite("debug_width_raw.png", width_roi)
        print(f"DEBUG legs ROI coords: y={legs_y1}:{legs_y2}, x={legs_x1}:{legs_x2}, shape={legs_roi.shape}")

    # ------------------------
    # OCR with appropriate preprocessing
    # ------------------------
    # Set debug_name only if debug is enabled
    dbg_prefix = lambda name: name if debug else None
    
    # Collect ROIs and run Tesseract (Otsu only)
    # We return the gray_scaled image for batch EasyOCR processing
    length_vals, length_counts, length_img = extract_numbers_from_roi(length_roi, rotate=False, remove_hatching=True, debug_name=dbg_prefix("length"), use_easyocr=True)
    height_vals, height_counts, height_img = extract_numbers_from_roi(height_roi, rotate=True, remove_hatching=False, debug_name=dbg_prefix("height"), use_easyocr=True)
    legs_vals, legs_counts, legs_img = extract_numbers_from_roi(legs_roi, rotate=True, remove_hatching=True, debug_name=dbg_prefix("legs"), use_easyocr=True)
    width_vals, width_counts, width_img = extract_numbers_from_roi(width_roi, rotate=True, remove_hatching=False, debug_name=dbg_prefix("width"), use_easyocr=True)

    # BATCH EASYOCR CALL (Much faster than 4 separate calls)
    try:
        reader = get_easyocr_reader()
        # readtext_batched returns a list of results, one for each image
        batch_results = reader.readtext_batched([length_img, height_img, legs_img, width_img], allowlist='0123456789')
        
        roi_data = [
            ("length", length_counts, length_vals),
            ("height", height_counts, height_vals),
            ("legs", legs_counts, legs_vals),
            ("width", width_counts, width_vals)
        ]
        
        for i, (name, counts, vals) in enumerate(roi_data):
            results = batch_results[i]
            for (bbox, text, prob) in results:
                if prob > 0.3:
                    nums = [int(x) for x in re.findall(r"\d{2,5}", text)]
                    for n in nums:
                        counts[n] += 3 # EasyOCR weight
                        if n not in vals:
                            vals.append(n)
                    if debug:
                        print(f"DEBUG {name} EasyOCR: OCR='{text}' nums={nums} conf={prob:.2f}")
    except Exception as e:
        if debug:
            print(f"DEBUG EasyOCR Batch Error: {e}")

    # ------------------------
    # Selection logic with domain filtering
    # AHU typical dimensions:
    # - Length: 1000-6000mm (1-6m)
    # - Width: 800-2500mm
    # - Height: 800-2500mm
    # - Legs: 150-600mm
    # ------------------------
    if length_vals:
        # Filter for realistic AHU length values (3000-5000mm typical)
        plausible_length = {v: length_counts[v] for v in length_vals if 3000 <= v <= 5000}
        
        if plausible_length:
            # Use consensus: prefer values that appear multiple times
            # Sort by count (descending), then by value (descending for largest)
            best_candidates = sorted(plausible_length.items(), key=lambda x: (-x[1], -x[0]))
            # Take the value with highest count (most agreement across OCR methods)
            result["length"] = best_candidates[0][0]
        else:
            # Fallback: try wider range (2000-6000)
            wider_range = {v: length_counts[v] for v in length_vals if 2000 <= v <= 6000}
            if wider_range:
                best_candidates = sorted(wider_range.items(), key=lambda x: (-x[1], -x[0]))
                result["length"] = best_candidates[0][0]

    if height_vals:
        # Filter for realistic height (800-2500mm typical)
        plausible_height = [v for v in height_vals if 600 <= v <= 3000]
        if plausible_height:
            result["height"] = max(plausible_height)
        else:
            result["height"] = max(height_vals)

    if legs_vals:
        # Filter for plausible leg values (typically 150-500 range)
        plausible_legs = [v for v in legs_vals if 100 <= v <= 600]
        if plausible_legs:
            # Legs are typically 200-400mm. If we have multiple values,
            # prefer values in the 200-400 range (common leg heights)
            preferred_legs = [v for v in plausible_legs if 200 <= v <= 400]
            if preferred_legs:
                result["legs"] = max(preferred_legs)  # e.g., 300
            else:
                # If no values in preferred range, take the smaller plausible value
                # (larger values like 500 are often misreads of 300)
                result["legs"] = min(plausible_legs)
    else:
        # Default legs value if OCR fails to detect
        result["legs"] = 300

    if width_vals:
        # Filter for realistic width (800-2500mm typical)
        plausible_width = [v for v in width_vals if 600 <= v <= 3000]
        if plausible_width:
            result["width"] = max(plausible_width)
        else:
            result["width"] = max(width_vals)

    return result

# =============================
# Post-formatting (important)
# =============================
def format_height_with_legs(height, legs):
    if height and legs:
        return f"{height} + {legs} Leg"
    if height:
        return str(height)
    return None

# ===============================
# FINAL OUTPUT CREATION
# ===============================
def normalize_filter_rating(text):
    if not text: return text
    # Normalize "TO" / "to" -> " down to "
    # Handle cases like "90%TO10MIC" or "90%to10mic"
    text = re.sub(r"(?i)TO", " down to ", text)
    
    # Normalize "MIC" / "MIC." -> "µ"
    text = re.sub(r"(?i)MIC\.?", "µ", text)
    
    # Clean up spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def create_final_output(parsed_data):
    """Combine all parsed data into the desired format."""
    final = {}
    
    # Title block fields
    if "title" in parsed_data:
        title = parsed_data["title"]
        final["system_no"] = title.get("system_no", "")
        
        # Model Normalization
        model = title.get("model", "")
        # AHUM-GN-DX2309-4726/13A -> AHUM-GN-DX2309-4726/13. Rev. A
        model_match = re.match(r"(.*)/(\d+)([A-Z])$", model)
        if model_match:
            final["model"] = f"{model_match.group(1)}/{model_match.group(2)}. Rev. {model_match.group(3)}"
        else:
            final["model"] = model
            
        final["manufacturer"] = title.get("manufacturer", "")
        
        # Description Normalization
        desc = title.get("description", "")
        # Check for "DOUBLE" and "SKIN" (even if concatenated like DOUBLESKIN)
        if "DOUBLE" in desc.upper() and "SKIN" in desc.upper():
            final["description"] = "Double skin F.M. AHU"
        else:
            final["description"] = desc
    
    # Additional fields
    if "additional" in parsed_data:
        for key, value in parsed_data["additional"].items():
            # Don't overwrite description if we already normalized it
            if key == "description" and final.get("description") == "Double skin F.M. AHU":
                continue
            final[key] = value
    
    # Filter make
    final["filter.make"] = parsed_data.get("filter_make", "AAF")
    
    # Filter fields
    filters = parsed_data.get("filters", {})
    
    for role in ["prefilter", "finefilter", "freshfilter", "bleedfilter"]:
        if role in filters:
            f_data = filters[role]
            
            # Class & Rating
            f_class = f_data.get("class", "")
            f_rating = f_data.get("rating", "")
            
            # Apply rating normalization
            if f_rating:
                f_rating = normalize_filter_rating(f_rating)
            else:
                # Defaults if missing
                if role == "prefilter" or role == "freshfilter":
                    f_rating = "90% down to 10µ"
                elif role == "finefilter":
                    f_rating = "75% down to 0.3µ"
                elif role == "bleedfilter":
                    f_rating = "99% down to 3µ"
            
            if f_class:
                final[f"{role}.class"] = f"{f_class} – ({f_rating})"
            
            # Type
            if f_data.get("type"):
                final[f"{role}.type"] = f_data["type"]
                
            # Size Qty
            if f_data.get("size_qty"):
                sizes = f_data["size_qty"]
                if len(sizes) > 1:
                    final[f"{role}.size_qty"] = '"' + '\n'.join(sizes) + '"'
                elif sizes:
                    final[f"{role}.size_qty"] = sizes[0]
    
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
        # Normalize "Water" to "water"
        c_type = cool.get("coil_type", "")
        if "Water" in c_type:
            c_type = c_type.replace("Water", "water")
        final["coil_type"] = c_type
        
        final["tube_material"] = cool.get("tube_material", "")
        if cool.get("fin_material"):
            final["fin_material"] = f"{cool['fin_material']} hydrophillic"
        final["fpi"] = str(cool.get("fpi", ""))
        final["rows"] = str(cool.get("rows", ""))
    
    if "heating_coil" in coils:
        heat = coils["heating_coil"]
        # Normalize "Water" to "water"
        h_type = heat.get("coil_type", "")
        if "Water" in h_type:
            h_type = h_type.replace("Water", "water")
        final["heating_coil_type"] = h_type
        
        final["heating_tube_material"] = heat.get("tube_material", "")
        if heat.get("fin_material"):
            final["heating_fin_material"] = f"{heat['fin_material']} hydrophillic"
        final["heating_fpi"] = str(heat.get("fpi", ""))
        final["heating_rows"] = str(heat.get("rows", ""))
    
    # Drain
    drain = parsed_data.get("drain", {})
    d_ins = drain.get("drain_insulation", "")
    # Normalize "Rubber" -> "Nitrile Rubber" if needed
    if "Rubber" in d_ins and "Nitrile" not in d_ins:
        d_ins = d_ins.replace("Rubber", "Nitrile Rubber")
    final["drain_insulation"] = d_ins
    
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
    import sys
    
    # Extract data from image
    image_path = "images/1f30e6cc-AHU-GF-06.png"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
    data = extract_text_by_region(image_path)
    
    # Extract dimensions
    dimensions = {}
    elevation_items = data.get("elevation_view", [])
    plan_items = data.get("plan_view", [])
    
    if elevation_items and plan_items:
        # Load image for dimension extraction
        img = cv2.imread(image_path)
        if img is not None:
            # Use the first detected box for each view
            elevation_box = elevation_items[0]["bbox"]
            plan_box = plan_items[0]["bbox"]
            
            # debug=False disables all dimension-related debug prints and image saves
            dimensions = extract_dimensions(img, elevation_box, plan_box, debug=False)

    # Parse all tables
    parsed_title = parse_title_block(data.get("title_block", []))
    parsed_filters, filter_make = parse_filter_table(data.get("filter_table", []))
    parsed_fan_motor = parse_fan_motor_table(data.get("fan_motor_table", []))
    parsed_coils = parse_coil_table(data.get("coil_table", []))
    parsed_drain = parse_drain_table(data.get("drain_table", []))
    parsed_damper = parse_damper_table(data.get("damper_table", []))
    parsed_additional = parse_additional_fields(data.get("title_block", []), data.get("filter_table", []), dimensions)
    
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
    
    # Print Raw OCR Data
    print("\n=== RAW OCR DATA ===")
    for k, v in data.items():
        print(f"\n{k.upper()}")
        print(k, ":", v)

    # Create final normalized output
    final_output = create_final_output(all_parsed)
    
    # Print Final Normalized Output
    print("\n=== FINAL OUTPUT ===")
    for key, value in final_output.items():
        if value:
            print(f"{key} : {value}")