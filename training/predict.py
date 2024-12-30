from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2


# load configs 
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = './output/model_final.pth'
cfg.MODEL.DEVICE = 'cpu'

# create the predictor
predictor = DefaultPredictor(cfg)

# load image
# image = cv2.imread("./data/val/imgs/00306_92_jpg.rf.28b9a088406c37a0100afef3082c2ca0.jpg")
image = cv2.imread(r"C:\Users\verta\Downloads\Tubes Viskom\training\data\data\val\imgs\00088_115_jpg.rf.26cac907f2a28eed111e445c71ef3515.jpg")

# prediction
outputs = predictor(image)

threshold = 0.5

# display result
preds = outputs["instances"].pred_classes.tolist()
scores = outputs["instances"].scores.tolist()
bboxes = outputs["instances"].pred_boxes

for j, bbox in enumerate(bboxes):
    bbox = bbox.tolist()

    score = scores[j]
    pred = preds[j]

    if score > threshold:
        x1, y1, x2, y2 = [int(i) for i in bbox]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

cv2.imshow('image', image)
cv2.waitKey(0)
