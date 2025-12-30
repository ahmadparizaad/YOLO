import re

def parse_filter_table(filter_items):
    """Parses filter table from OCR output with proper table structure."""
    lines = []
    for item in filter_items:
        for text, _ in flatten_ocr_text(item["text"]):
            t = re.sub(r"\s+", " ", text.upper()).strip()
            if t:
                lines.append(t)
    
    # Parse filter manufacturer
    filter_make = "AAF"
    for line in lines:
        if "AAF" in line or "AAR" in line:  # Handle OCR errors
            filter_make = "AAF"
            break
    
    # Initialize result structure
    filter_mapping = {
        "prefilter": {"class": None, "type": None, "sizes": []},
        "freshfilter": {"class": None, "type": None, "sizes": []},
        "finefilter": {"class": None, "type": None, "sizes": []},
        "bleedfilter": {"class": None, "type": None, "sizes": []}
    }
    
    # Process each line
    for line in lines:
        # Skip header lines
        if "FILTER DETAILS" in line or "TYPE" in line or "RATING" in line or "SIZE" in line or "QTY" in line:
            continue
        
        # Determine filter type
        filter_type = None
        if "PREFILTER" in line or "PRE FILTER" in line:
            filter_type = "PRE FILTER"
        elif "SEMIHEPA" in line or "SEMI HEPA" in line:
            filter_type = "SEMI HEPA"
        elif "FINE FILTER" in line or "FINEFILTER" in line:
            filter_type = "FINE FILTER"
        elif "HEPA FILTER" in line or "HEPAFILTER" in line:
            filter_type = "HEPA FILTER"
        
        if not filter_type:
            continue
        
        # Extract filter class
        class_match = re.search(r'\((G-?\d+|F-?\d+|H-?\d+|U-?\d+)\)', line)
        if not class_match:
            class_match = re.search(r'(G-?\d+|F-?\d+|H-?\d+|U-?\d+)', line)
        
        filter_class = class_match.group(1) if class_match else None
        if filter_class and '-' not in filter_class and len(filter_class) > 1:
            # Add dash if missing (e.g., G4 -> G-4)
            filter_class = f"{filter_class[0]}-{filter_class[1:]}"
        
        # Check if it has holes
        has_holes = "WITH HOLES" in line or "WITHHOLES" in line
        
        # Determine functional role
        if filter_type == "PRE FILTER":
            role = "freshfilter" if has_holes else "prefilter"
        elif filter_type == "FINE FILTER":
            role = "bleedfilter" if has_holes else "finefilter"
        elif filter_type == "SEMI HEPA":
            role = "finefilter"  # Semi HEPA always goes to finefilter
        elif filter_type == "HEPA FILTER":
            role = "bleedfilter"
        else:
            continue
        
        # Set filter class if found
        if filter_class and not filter_mapping[role]["class"]:
            filter_mapping[role]["class"] = filter_class
        
        # Check if it's flange type
        if "FLANGE TYPE" in line or "FLANGETYPE" in line or "PREwu TYPE" in line:
            filter_mapping[role]["type"] = "Flange Type"
        
        # Extract all size patterns from this line
        # Pattern: H-610XW-610XD-50 or H-610 X W-610 X D-50
        # Also handles patterns like "RESTS XW-610XD-3001" where H- is missing
        
        # First, try standard pattern with H-
        size_patterns = list(re.finditer(
            r'H-?(\d{3})\s*X?\s*W-?(\d{3})\s*X?\s*D-?(\d{2,3})(\d)?',
            line
        ))
        
        # If no matches, try pattern without H- (starts with W-)
        if not size_patterns:
            size_patterns = list(re.finditer(
                r'W-?(\d{3})\s*X?\s*W-?(\d{3})\s*X?\s*D-?(\d{2,3})(\d)?',
                line
            ))
            # In this case, treat first match as W and second as H (reversed)
            for match in size_patterns:
                w = match.group(1)
                h = match.group(2)
                d = match.group(3)
                trailing_digit = match.group(4)
                qty = trailing_digit if trailing_digit else "1"
                formatted_size = f"{w} x {h} x {d} – {qty.zfill(2)} Nos."
                if formatted_size not in filter_mapping[role]["sizes"]:
                    filter_mapping[role]["sizes"].append(formatted_size)
        else:
            # Standard pattern found
            for match in size_patterns:
                h = match.group(1)
                w = match.group(2)
                d = match.group(3)
                trailing_digit = match.group(4)
                qty = trailing_digit if trailing_digit else "1"
                formatted_size = f"{w} x {h} x {d} – {qty.zfill(2)} Nos."
                if formatted_size not in filter_mapping[role]["sizes"]:
                    filter_mapping[role]["sizes"].append(formatted_size)
    
    # Convert to final result format
    result = {}
    for role, data in filter_mapping.items():
        if data["class"] or data["sizes"]:
            result[role] = {
                "class": data["class"],
                "type": data["type"],
                "size_qty": data["sizes"]
            }
    
    return result, filter_make


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
    
    # Helper to add efficiency suffix
    def get_efficiency_suffix(filter_class):
        if not filter_class:
            return ""
        if "G-4" in filter_class:
            return " – (90% down to 10µ)"
        elif "F-9" in filter_class or "F9" in filter_class:
            return " – (75% down to 0.3µ)"
        elif "F-7" in filter_class or "F7" in filter_class:
            return " – (99% down to 3µ)"
        return ""
    
    # Prefilter
    if "prefilter" in filters:
        pref = filters["prefilter"]
        if pref.get("class"):
            final["prefilter.class"] = f"{pref['class']}{get_efficiency_suffix(pref['class'])}"
        if pref.get("type"):
            final["prefilter.type"] = pref["type"]
        if pref.get("size_qty"):
            sizes = pref["size_qty"]
            if len(sizes) > 1:
                final["prefilter.size_qty"] = '"' + ' \n'.join(sizes) + ' "'
            elif sizes:
                final["prefilter.size_qty"] = sizes[0]
    
    # Finefilter
    if "finefilter" in filters:
        fine = filters["finefilter"]
        if fine.get("class"):
            final["finefilter.class"] = f"{fine['class']}{get_efficiency_suffix(fine['class'])}"
        if fine.get("type"):
            final["finefilter.type"] = fine["type"]
        if fine.get("size_qty"):
            sizes = fine["size_qty"]
            if len(sizes) > 1:
                final["finefilter.size_qty"] = '"' + '\n'.join(sizes) + '"'
            elif sizes:
                final["finefilter.size_qty"] = sizes[0]
    
    # Freshfilter
    if "freshfilter" in filters:
        fresh = filters["freshfilter"]
        if fresh.get("class"):
            final["freshfilter.class"] = f"{fresh['class']}{get_efficiency_suffix(fresh['class'])}"
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
            # Remove dash for F7 in bleedfilter
            display_class = bleed["class"].replace("-", "") if "F-7" in bleed["class"] or "F7" in bleed["class"] else bleed["class"]
            final["bleedfilter.class"] = f"{display_class}{get_efficiency_suffix(bleed['class'])}"
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
