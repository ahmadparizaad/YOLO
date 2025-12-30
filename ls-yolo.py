# Map Label Studio labels → YOLO region classes
LABEL_GROUPS = {
    "title_block": {
        "description", "system_no", "manufacturer", "model"
    },
    "filter_table": {
        "filter_entity", "filter_class", "filter_type",
        "filter_make", "filter_size_qty"
    },
    "fan_motor_table": {
        "fan_details", "fan_row", "motor_row", "col_make"
    },
    "coil_table": {
        "cooling_coil", "heating_coil",
        "coil_type", "tube_material",
        "fin_material", "fpi", "rows"
    },
    "drain_table": {
        "drain_insulation", "drain_mist_eliminator"
    },
    "damper_table": {
        "damper_row", "damper_type", "damper_size"
    },
    "elevation_view": {"elevation_view"},
    "plan_view": {"plan_view"}
}

YOLO_CLASSES = list(LABEL_GROUPS.keys())
CLASS_TO_ID = {name: idx for idx, name in enumerate(YOLO_CLASSES)}

import json
import os
from collections import defaultdict
from PIL import Image

LS_JSON_PATH = "project-8-at-2025-12-29-15-17-b5aa8013.json"
IMAGE_ROOT = "./images"        # folder containing images
OUTPUT_LABELS = "./labels"     # YOLO .txt output

os.makedirs(OUTPUT_LABELS, exist_ok=True)

def merge_boxes(boxes):
    """
    boxes: list of (x1, y1, x2, y2) in absolute pixels
    """
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return x1, y1, x2, y2

with open(LS_JSON_PATH, "r") as f:
    tasks = json.load(f)

for task in tasks:
    if not task.get("annotations"):
        continue

    image_path = os.path.basename(task["data"]["image"])
    image_file = os.path.join(IMAGE_ROOT, image_path)

    if not os.path.exists(image_file):
        print(f"Missing image: {image_file}")
        continue

    img = Image.open(image_file)
    img_w, img_h = img.size

    region_boxes = defaultdict(list)

    for ann in task["annotations"]:
        for res in ann.get("result", []):
            if res["type"] != "rectanglelabels":
                continue

            label = res["value"]["rectanglelabels"][0]

            for yolo_class, ls_labels in LABEL_GROUPS.items():
                if label in ls_labels:
                    x = res["value"]["x"] / 100 * img_w
                    y = res["value"]["y"] / 100 * img_h
                    w = res["value"]["width"] / 100 * img_w
                    h = res["value"]["height"] / 100 * img_h

                    region_boxes[yolo_class].append(
                        (x, y, x + w, y + h)
                    )

    if not region_boxes:
        continue

    label_path = os.path.join(
        OUTPUT_LABELS,
        os.path.splitext(image_path)[0] + ".txt"
    )

    with open(label_path, "w") as f:
        for cls, boxes in region_boxes.items():
            x1, y1, x2, y2 = merge_boxes(boxes)

            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h

            f.write(
                f"{CLASS_TO_ID[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
            )

print("Conversion completed.")
