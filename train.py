from ultralytics import YOLO

# YOLO model load
model = YOLO("yolo11n.pt")

# Model training
model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=8
)
