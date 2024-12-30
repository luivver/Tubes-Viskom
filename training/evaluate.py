import os
import argparse
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo  # Add this import
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import load_coco_json

# Register the dataset
DatasetCatalog.register(
    "val_dataset",
    lambda: load_coco_json(
        r"C:\Users\verta\Downloads\Tubes Viskom\training\data\data\val\annotations.json",  # Correct path
        r"C:\Users\verta\Downloads\Tubes Viskom\training\data\data\val\imgs",  # Correct path
        "val_dataset",
    ),
)

# Verify registration
val_metadata = MetadataCatalog.get("val_dataset")
print("Dataset registered:", val_metadata)

def main():
    # Load the model config and weights
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = './output/model_final.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for predictions
    cfg.MODEL.DEVICE = 'cuda'
    
    # Specify the number of classes
    cfg.MODEL.RETINANET.NUM_CLASSES = 2  # Replace with the number of classes in your dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Only needed if using ROI-based models like Faster R-CNN


    # Ensure dataset is registered
    evaluator = COCOEvaluator("val_dataset", cfg, False, output_dir="./eval_output")
    val_loader = build_detection_test_loader(cfg, "val_dataset")

    # Run evaluation
    print("Running evaluation...")
    results = inference_on_dataset(DefaultPredictor(cfg).model, val_loader, evaluator)
    print("Evaluation results:")
    print(results)

if __name__ == "__main__":
    main()
