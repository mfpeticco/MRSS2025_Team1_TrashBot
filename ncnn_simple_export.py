from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO('models/yolov11n_trash_detection/weights/best.pt')

# Export the model to NCNN format
model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'