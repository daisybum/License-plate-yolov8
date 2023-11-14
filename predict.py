from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(model='runs\\detect\\train3\\weights\\platev8n.pt')  # build a new model from scratch

    # Use the model
    results = model.predict('data\\demo.mp4', save=True, imgsz=640, conf=0.01)