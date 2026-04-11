from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,   # 🔥 مهم جداً
        name="cones_model",
        project="runs/detect"
    )

if __name__ == "__main__":
    freeze_support()  # 🔥 الحل
    main()