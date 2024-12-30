import os
from PIL import Image
import json

# Paths
images_dir = r"C:\Users\verta\Downloads\Tubes Viskom\training\data\data\val\imgs"
annotations_path = r"C:\Users\verta\Downloads\Tubes Viskom\training\data\data\val\annotations.json"

# Load annotations
with open(annotations_path, "r") as f:
    data = json.load(f)

# Check image dimensions
for img_info in data["images"]:
    img_path = os.path.join(images_dir, img_info["file_name"])
    if os.path.exists(img_path):
        with Image.open(img_path) as img:
            actual_width, actual_height = img.size
        if (img_info["width"], img_info["height"]) != (actual_width, actual_height):
            print(f"Mismatch for {img_info['file_name']}:")
            print(f"  Annotation: {img_info['width']}x{img_info['height']}")
            print(f"  Actual: {actual_width}x{actual_height}")
            # Update JSON
            img_info["width"], img_info["height"] = actual_width, actual_height
    else:
        print(f"Image not found: {img_info['file_name']}")

# Save corrected annotations
with open(annotations_path, "w") as f:
    json.dump(data, f)

print("Finished checking and updating image dimensions.")
