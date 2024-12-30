import os
import json

def yolo_to_coco(yolo_annotations_dir, output_coco_json, image_dir):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Define your categories (adjust as necessary)
    categories = [{"id": 1, "name": "category_name"}]
    coco_format["categories"] = categories

    annotation_id = 0

    # Loop through annotation files
    for idx, annotation_file in enumerate(os.listdir(yolo_annotations_dir)):
        # Define a unique image ID for each image
        image_id = os.path.splitext(annotation_file)[0]
        image_path = os.path.join(image_dir, f"{image_id}.jpg")

        # Ensure the corresponding image file exists
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found! Skipping...")
            continue

        # Get image dimensions (replace with actual image size or use a library like OpenCV)
        # If you want to dynamically get image size, uncomment below:
        # import cv2
        # img = cv2.imread(image_path)
        # height, width, _ = img.shape

        height, width = 720, 1280  # Replace with actual dimensions

        # Add image information to COCO format
        coco_format["images"].append({
            "id": idx,  # Unique integer ID
            "file_name": f"{image_id}.jpg",
            "width": width,
            "height": height
        })

        # Read YOLO annotation
        annotation_path = os.path.join(yolo_annotations_dir, annotation_file)
        with open(annotation_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                category_id = int(parts[0]) + 1  # YOLO categories are 0-indexed, COCO is 1-indexed
                x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

                # Convert YOLO format to COCO format
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                bbox_width = bbox_width * width
                bbox_height = bbox_height * height

                # Add annotation information to COCO format
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": idx,  # Link to the corresponding image
                    "category_id": category_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                annotation_id += 1

    # Save to output JSON file
    with open(output_coco_json, "w") as json_file:
        json.dump(coco_format, json_file)
    print(f"COCO JSON file created at: {output_coco_json}")

# Example usage
if __name__ == "__main__":
    yolo_annotations_dir = r"C:\Users\verta\Downloads\Tubes Viskom\training\data\data\val\anns"
    output_coco_json = r"C:\Users\verta\Downloads\Tubes Viskom\training\data\data\val\annotations.json"
    image_dir = r"C:\Users\verta\Downloads\Tubes Viskom\training\data\data\val\imgs"
    yolo_to_coco(yolo_annotations_dir, output_coco_json, image_dir)