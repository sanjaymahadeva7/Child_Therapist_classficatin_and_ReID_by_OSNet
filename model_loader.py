from ultralytics import YOLO

def load_model(model_path='best.pt'):
    # Load the YOLOv8 model
    model = YOLO(model_path)
    return model
