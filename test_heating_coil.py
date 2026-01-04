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

def test_heating_coil():
    fields = ["heating_coil_type", "heating_tube_material", "heating_fin_material", "heating_fpi", "heating_rows"]
    header = f"{'Image Path':<60} | " + " | ".join([f"{f:<20}" for f in fields])
    print(header)
    print("-" * len(header))
    
    for img_path in TEST_IMAGES:
        if not os.path.exists(img_path):
            print(f"{img_path:<60} | FILE NOT FOUND")
            continue
            
        try:
            data = pipeline.extract_text_by_region(img_path)
            parsed_coils = pipeline.parse_coil_table(data.get("coil_table", []))
            
            # Use create_final_output to get normalized fields
            all_parsed = {"coils": parsed_coils}
            final_output = pipeline.create_final_output(all_parsed)
            
            row = f"{img_path:<60} | "
            row += " | ".join([f"{str(final_output.get(f, '')): <20}" for f in fields])
            print(row)
        except Exception as e:
            print(f"{img_path:<60} | ERROR: {str(e)}")

if __name__ == "__main__":
    test_heating_coil()
