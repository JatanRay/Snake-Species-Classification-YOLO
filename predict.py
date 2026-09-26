from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

results = model.predict(
    source="dataset/images/val/84K9PASMKSUR_jpg.rf.0c29acd5cd24f06f2b74022fdd1a966d.jpg",
    conf=0.5
)

for result in results:
    image = result.plot()

    cv2.imwrite("snake_result.jpg", image)

print("Prediction complete!")