import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from ultralytics import YOLO


def detect_license_plate(plate_model, car_image, attempt=1, conf=0.5):
    # Perform license plate detection
    plate_results = plate_model(car_image, conf=conf, verbose=False)[0].boxes
    plate_box = plate_results[plate_results.cls == 0].xyxy  # Bounding boxes

    # If no plates detected, and we're on the first attempt, lower the threshold and try again
    if len(plate_box) == 0 and attempt == 1:
        plate_model.conf = 0.1  # Lowering the confidence threshold
        return detect_license_plate(plate_model, car_image, attempt=2, conf=0.1)

    # Reset the confidence threshold after the second attempt
    if len(plate_box) == 0 and attempt == 2:
        return detect_license_plate(plate_model, car_image, attempt=3, conf=0.01)

    return plate_box


def detect_cars_and_plates(car_model, plate_model, image_path):
    # Load image
    img = Image.open(image_path)

    # Perform car detection
    results = car_model(img, verbose=False)[0].boxes
    car_results = results[(results.cls == 2) | (results.cls == 7)]
    car_boxes = car_results.xyxy.cpu().numpy()
    car_clses = np.expand_dims(car_results.cls.cpu().numpy(), axis=1)
    car_bboxs = np.hstack((car_clses, car_boxes))  # xyxy is the bounding box format (x1, y1, x2, y2)

    # Iterate over detected cars
    for box in car_boxes:
        x1, y1, x2, y2 = map(int, box)
        # Crop the car image
        car_img = img.crop((x1, y1, x2, y2))

        # Try to detect a license plate in the car image
        plate_box = detect_license_plate(plate_model, car_img)

        # If a plate was detected
        if len(plate_box) > 0:
            bbox = plate_box[0].cpu().numpy()
            draw = ImageDraw.Draw(img)
            bbox[[0, 2]] += x1
            bbox[[1, 3]] += y1
            bbox = np.hstack((np.zeros((1)), bbox))
            # draw.rectangle(bbox, outline=(255, 0, 0), width=3)

            print(f"License plate detected at: {bbox}")
            plate_box = np.expand_dims(bbox, axis=0)
            car_bboxs = np.vstack((car_bboxs, plate_box))

        else:
            print(f"No license plate detected for car at: {box}, even after second attempt.")

    return car_bboxs

if __name__ == "__main__":
    # Example usage
    # Load the models
    car_model = YOLO(model='model\\yolov8x.pt')
    plate_model = YOLO(model='model\\platev8n.pt')
    detect_cars_and_plates(car_model, plate_model, 'data\\demo.jpg')
