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

import cv2
import os

IMAGE_DIR = "images"
LABEL_DIR = "labels"
OUTPUT_DIR = "vis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

COLORS = [
    (255, 0, 0),    # blue
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 0, 128),
    (0, 128, 255),
    (128, 128, 0),
]

def draw_boxes(image_path, label_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    if not os.path.exists(label_path):
        return img

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        cls, xc, yc, bw, bh = map(float, line.split())
        cls = int(cls)

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        color = COLORS[cls % len(COLORS)]
        label = CLASSES[cls]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return img

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    lbl_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

    vis_img = draw_boxes(img_path, lbl_path)

    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, vis_img)

print("Visualization complete. Check the 'vis/' folder.")
