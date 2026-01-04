import os
import cv2
from ultralytics import YOLO
import ocr_pipeline2 as pipeline

TEST_IMAGES = [
    "images/005268f8-13_3868_CMH-2275_CFM_FM_AHU-GF-15-Model_page_001.png",
    "images/06f03d09-08_6885_CMH-4050_CFM_FM_AHU-GF-18-Model_page_001.png",
    "images/09676ba7-11_8347_CMH-4910_CFM_FM_AHU-GF-08-Model_page_001.png",
    "images/12c83249-05_24939_CMH-14670_CFM_FM_AHU-GF-06-Model_page_001.png",
    "images/16f6cc23-06_8543_CMH-5025_CFM_FM_AHU-GF-07-Model_page_001.png",
    "images/57770232-12_22168_CMH-13040_CFM_FM_AHU-GF-12-Model_page_001.png",
    "images/5a484802-15_7174_CMH-4220_CFM_FM_AHU-GF-16-Model_page_001.png",
    "images/618c40ed-14_12342_CMH-7260_CFM_FM_AHU-GF-11-Model_page_001.png",
    "images/6702ecf8-04_19550_CMH-11500_CFM_FM_AHU-GF-04-Model_page_001.png",
    "images/8b360c88-20_8755_CMH-5150_CFM_FM_AHU-GF-09-Model_page_001.png"
]

def test_system_no():
    print(f"{'Image Path':<80} | {'System No':<20}")
    print("-" * 105)
    for img_path in TEST_IMAGES:
        if not os.path.exists(img_path):
            print(f"{img_path:<80} | FILE NOT FOUND")
            continue
            
        try:
            data = pipeline.extract_text_by_region(img_path)
            parsed_title = pipeline.parse_title_block(data.get("title_block", []))
            system_no = parsed_title.get("system_no", "NOT FOUND")
            print(f"{img_path:<80} | {system_no:<20}")
        except Exception as e:
            print(f"{img_path:<80} | ERROR: {str(e)}")

if __name__ == "__main__":
    test_system_no()
