import os
from glob import glob

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from utils import detect_cars_and_plates

DATA_DIR = "D:\\number_plate_dataset\\images"
img_paths = glob(os.path.join(DATA_DIR, "*\\*.jpg"))

car_model = YOLO('model\\yolov8x.pt')
plate_model = YOLO('model\\platev8n.pt')

for path in tqdm(img_paths):
    label_path = path.replace('images', 'labels').replace('jpg', 'txt')
    bounding_boxes = detect_cars_and_plates(car_model, plate_model, path)
    np.savetxt(label_path, bounding_boxes)
