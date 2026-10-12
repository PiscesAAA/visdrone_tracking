from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("models/yolov8s-p2-bifpn.yaml").load("yolov8s.pt")

    model.train(
        data="configs/VisDrone.yaml",
        epochs=150,
        imgsz=960,
        batch=2,
        patience=30,
        degrees=15.0,
        scale=0.5,
        mosaic=1.0,
        optimizer="AdamW",
        lr0=0.001,
        max_det=1000,
        device=0,
        workers=4,
        seed=42,
        project="runs",
        name="p2_pretrained",
        exist_ok=True
    )
