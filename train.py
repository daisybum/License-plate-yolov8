from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8x.yaml")  # build a new model from scratch

    # Use the model
    results = model.train(data="configs\\config.yaml", epochs=20, verbose=True, device=0)  # train the model
    # yolo task=detect mode=train data="configs.yaml" epochs=20 verbose=True
