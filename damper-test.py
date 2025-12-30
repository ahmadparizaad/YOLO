import re

def parse_damper_table(damper_items):
    """
    Parses damper sizes from OCR output in tabular format.
    Handles tables with columns: TYPE, MATERIAL, SIZE, QTY.
    """
    
    lines = []
    
    # Flatten OCR text
    for item in damper_items:
        for text, _ in item["text"]:
            t = re.sub(r"\s+", " ", text.upper()).strip()
            lines.append(t)
    
    result = {
        "damper_supply_air": None,
        "damper_return_air": None,
        "damper_fresh_air": None,
        "damper_bleed_air": None,  # Note: OCR says EXHAUST AIR DAMPER, not BLEED
        "damper_coil_bypass": None
    }
    
    # Process each line looking for damper entries
    for line in lines:
        # Skip header lines and empty lines
        if not line or "OPENING DETAILS" in line or "TYPE" in line or "MATERIAL" in line:
            continue
        
        # Check if this line contains a damper entry
        # Look for damper type indicators
        role = None
        
        if "SUPPLY" in line and ("AIR" in line or "DAMPER" in line):
            role = "damper_supply_air"
        elif "RETURN" in line and ("AIR" in line or "DAMPER" in line):
            role = "damper_return_air"
        elif "FRESH" in line and ("AIR" in line or "DAMPER" in line):
            role = "damper_fresh_air"
        elif ("EXHAUST" in line or "BLEED" in line) and ("AIR" in line or "DAMPER" in line):
            # Note: OCR says EXHAUST, target says BLEED - treat as same
            role = "damper_bleed_air"
        elif "BYPASS" in line and ("AIR" in line or "DAMPER" in line):
            role = "damper_coil_bypass"
        
        if not role:
            continue
        
        # Extract dimensions - multiple patterns to handle different formats
        size_match = None
        
        # Pattern 1: W—400 X H-400 (with em dash or regular dash)
        if not size_match:
            size_match = re.search(r"W[—\-]\s*(\d{2,4})\s*[X×]\s*H[—\-]\s*(\d{2,4})", line)
        
        # Pattern 2: W 400 X H 400 (with spaces)
        if not size_match:
            size_match = re.search(r"W\s*(\d{2,4})\s*[X×]\s*H\s*(\d{2,4})", line)
        
        # Pattern 3: Generic dimension pattern (might catch quantity too, so be careful)
        if not size_match:
            # Look for patterns like 400 X 350
            size_match = re.search(r"(\d{3})\s*[X×]\s*(\d{3})", line)
        
        if size_match:
            width, height = size_match.groups()
            # Format consistently
            result[role] = f"{width} x {height}"
        else:
            result[role] = None
    
    return result


# Alternative approach: Parse table rows more systematically
def parse_damper_table_structured(damper_items):
    """
    Alternative approach that tries to parse the table structure more explicitly.
    """
    
    # First, extract all text
    all_text = []
    for item in damper_items:
        for text, _ in item["text"]:
            all_text.append(text.upper().strip())
    
    # Join all text to see the full content
    full_text = " ".join(all_text)
    
    result = {
        "damper_supply_air": None,
        "damper_return_air": None,
        "damper_fresh_air": None,
        "damper_bleed_air": None,
        "damper_coil_bypass": None
    }
    
    # Define patterns for each damper type
    patterns = {
        "damper_supply_air": r"SUPPLY\s*(?:AIR\s*)?DAMPER[^X]*W[—\-]\s*(\d{3})\s*[X×]\s*H[—\-]\s*(\d{3})",
        "damper_return_air": r"RETURN\s*(?:AIR\s*)?DAMPER[^X]*W[—\-]\s*(\d{3})\s*[X×]\s*H[—\-]\s*(\d{3})",
        "damper_fresh_air": r"FRESH\s*(?:AIR\s*)?DAMPER[^X]*W[—\-]\s*(\d{3})\s*[X×]\s*H[—\-]\s*(\d{3})",
        "damper_bleed_air": r"(?:EXHAUST|BLEED)\s*(?:AIR\s*)?DAMPER[^X]*W[—\-]\s*(\d{3})\s*[X×]\s*H[—\-]\s*(\d{3})",
        "damper_coil_bypass": r"BYPASS\s*(?:AIR\s*)?DAMPER[^X]*W[—\-]\s*(\d{3})\s*[X×]\s*H[—\-]\s*(\d{3})",
    }
    
    # Try to extract each damper
    for role, pattern in patterns.items():
        match = re.search(pattern, full_text)
        if match:
            width, height = match.groups()
            result[role] = f"{width} x {height}"
    
    return result


# Even better: Parse row by row if OCR preserves row structure
def parse_damper_table_row_based(damper_items):
    """
    Assumes OCR preserves some row structure with pipe separators or newlines.
    """
    
    # Extract text with bounding box info if available
    rows = []
    for item in damper_items:
        row_texts = []
        for text, bbox in item["text"]:
            row_texts.append(text.strip())
        if row_texts:
            rows.append(" ".join(row_texts).upper())
    
    result = {
        "damper_supply_air": None,
        "damper_return_air": None,
        "damper_fresh_air": None,
        "damper_bleed_air": None,
        "damper_coil_bypass": None
    }
    
    # Map damper types
    damper_map = {
        "SUPPLY": "damper_supply_air",
        "RETURN": "damper_return_air", 
        "FRESH": "damper_fresh_air",
        "EXHAUST": "damper_bleed_air",  # OCR says EXHAUST, target says BLEED
        "BLEED": "damper_bleed_air",
        "BYPASS": "damper_coil_bypass"
    }
    
    for row in rows:
        # Skip header rows
        if not row or "OPENING" in row or "TYPE" in row or "MATERIAL" in row or "SIZE" in row:
            continue
        
        # Check which damper type this row contains
        role = None
        for damper_key, result_key in damper_map.items():
            if damper_key in row:
                role = result_key
                break
        
        if not role:
            continue
        
        # Extract dimensions - handle the format W—400 X H-400
        size_match = re.search(r"W[—\-]\s*(\d{3})\s*[X×]\s*H[—\-]\s*(\d{3})", row)
        if size_match:
            width, height = size_match.groups()
            result[role] = f"{width} x {height}"
        else:
            # Try alternative pattern
            size_match = re.search(r"(\d{3})\s*[X×]\s*(\d{3})", row)
            if size_match:
                width, height = size_match.groups()
                result[role] = f"{width} x {height}"
    
    return result


# Test with simulated OCR data
if __name__ == "__main__":
    # Simulate the table OCR based on your image
    test_ocr = [
        {"text": [
            ("| 6 | OPENING DETAILS:", {}),
            ("|", {}),
            ("|", {}),
            ("TYPE", {}),
            ("MATERIAL", {}),
            ("SIZE", {}),
            ("QTY.", {})
        ]},
        {"text": [
            ("A", {}),
            ("SUPPLY AIR DAMPER", {}),
            ("ALUMINIUM", {}),
            ("W—400 X H-400", {}),
            ("1", {})
        ]},
        {"text": [
            ("B", {}),
            ("RETURN AIR DAMPER", {}),
            ("ALUMINIUM", {}),
            ("W—400 X H-350", {}),
            ("1", {})
        ]},
        {"text": [
            ("C", {}),
            ("FRESH AIR DAMPER", {}),
            ("ALUMINIUM", {}),
            ("W—560 X H-255", {}),
            ("1", {})
        ]},
        {"text": [
            ("D", {}),
            ("EXHAUST AIR DAMPER", {}),
            ("ALUMINIUM", {}),
            ("W—295 X H-315", {}),
            ("1", {})
        ]},
        {"text": [
            ("E", {}),
            ("BYPASS AIR DAMPER", {}),
            ("ALUMINIUM", {}),
            ("W—879 X H-115", {}),
            ("1", {})
        ]}
    ]
    
    print("Testing parse_damper_table:")
    result1 = parse_damper_table(test_ocr)
    print(result1)
    
    print("\nTesting parse_damper_table_structured:")
    result2 = parse_damper_table_structured(test_ocr)
    print(result2)
    
    print("\nTesting parse_damper_table_row_based:")
    result3 = parse_damper_table_row_based(test_ocr)
    print(result3)