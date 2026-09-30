import os
from pathlib import Path
import cv2

# 数据集根目录，根据自己的路径调整
ROOT_DIR = Path("datasets/VisDrone")


def convert_visdrone_to_yolo(split):
    img_dir = ROOT_DIR / f"VisDrone2019-DET-{split}" / "images"
    anno_dir = ROOT_DIR / f"VisDrone2019-DET-{split}" / "annotations"
    out_dir = ROOT_DIR / f"VisDrone2019-DET-{split}" / "labels"

    # 创建 labels 文件夹
    out_dir.mkdir(parents=True, exist_ok=True)

    if not anno_dir.exists():
        print(f"警告：找不到 {anno_dir}，请检查路径和数据集是否解压正确！")
        return

    for anno_file in anno_dir.glob("*.txt"):
        img_file = img_dir / (anno_file.stem + ".jpg")
        if not img_file.exists():
            continue

        # 读取图片获取宽高
        img = cv2.imread(str(img_file))
        if img is None: continue
        h, w = img.shape[:2]

        with open(anno_file, "r") as f, open(out_dir / anno_file.name, "w") as out_f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6: continue         #防止遇到少于6个字段的，从而导致报错

                x, y, bw, bh, score, category = map(int, parts[:6])

                # VisDrone中 category 为 0 是忽略区域，score 为 0 表示无效
                if category == 0 or score == 0: continue

                # VisDrone类别 1~10 对应 YOLO 类别 0~9
                cls_id = category - 1

                # 计算归一化中心点和宽高
                x_c = (x + bw / 2) / w
                y_c = (y + bh / 2) / h
                norm_w = bw / w
                norm_h = bh / h

                # 限制在 0-1 之间，防止浮点误差越界
                x_c, y_c = max(0, min(1, x_c)), max(0, min(1, y_c))
                norm_w, norm_h = max(0, min(1, norm_w)), max(0, min(1, norm_h))

                out_f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    print(f"✅ {split} 转换完成！保存至 {out_dir}")


if __name__ == "__main__":
    for split in ["train", "val", "test-dev"]:
        convert_visdrone_to_yolo(split)