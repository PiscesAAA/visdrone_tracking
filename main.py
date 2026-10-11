from ultralytics import YOLO

if __name__ == "__main__":

    # 因为没有预训练权重，需要从 yaml 从头开始训练
    model = YOLO("models/yolov8s-p2-bifpn.yaml")

    model.train(
        data="configs/VisDrone.yaml",
        epochs=100,
        imgsz=960,
        batch=2,
        patience=20,
        degrees=15.0,
        scale=0.5,
        mosaic=1.0,
        optimizer="AdamW",
        lr0=0.001,
        max_det=1000,
        device=0,
        workers=4,  # P2层显存占用变大，workers设小一点
        seed=42,
        project="runs",
        name="p2_baseline",  # 调整后的新名字
        exist_ok=True
    )
