import colorsys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox

_PROJECT_DIR = Path(__file__).parent
_MODEL_DIR = _PROJECT_DIR / "models"


class Detector:
    """YOLOv7 目标检测器封装，接口兼容原有 pipeline"""

    def __init__(
        self,
        model_path: str = "ep950-loss0.050-val_loss0.055.pth",
        classes_path: str | None = None,
        anchors_path: str | None = None,
        input_shape: tuple = (640, 640),
        phi: str = "l",
        anchors_mask: list | None = None,
        letterbox: bool = True,
        device: str = "auto",
    ):
        if anchors_mask is None:
            anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        if classes_path is None:
            classes_path = str(_PROJECT_DIR / "model_data" / "coco_classes.txt")
        if anchors_path is None:
            anchors_path = str(_PROJECT_DIR / "model_data" / "yolo_anchors.txt")

        self.input_shape = input_shape
        self.letterbox = letterbox

        if device == "cuda":
            self.cuda = torch.cuda.is_available()
            if not self.cuda:
                raise RuntimeError("CUDA 不可用，请切换为 auto 或 cpu")
        elif device == "cpu":
            self.cuda = False
        else:
            self.cuda = torch.cuda.is_available()
        self.device_str = device

        self.class_names, self.num_classes = get_classes(classes_path)
        self.anchors, self.num_anchors = get_anchors(anchors_path)
        self.bbox_util = DecodeBox(
            self.anchors, self.num_classes, input_shape, anchors_mask
        )

        # Color generation per class
        hsv_tuples = [(x / self.num_classes, 1.0, 1.0)
                      for x in range(self.num_classes)]
        colors = [colorsys.hsv_to_rgb(*t) for t in hsv_tuples]
        self.colors = [
            (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors
        ]

        self.net = YoloBody(anchors_mask, self.num_classes, phi)
        torch_device = torch.device("cuda" if self.cuda else "cpu")
        model_path_resolved = str(_MODEL_DIR / model_path)
        self.net.load_state_dict(
            torch.load(model_path_resolved, map_location=torch_device)
        )
        self.net = self.net.fuse().eval()
        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray, conf: float = 0.25, nms_iou: float = 0.3):
        """
        image: RGB 格式 np.ndarray (H, W, 3)
        conf:  置信度阈值
        nms_iou: NMS IoU 阈值

        返回: (annotated_image, detections)
            annotated_image: 已绘制检测框的 RGB 图片
            detections: [{label, confidence, bbox}, ...]
        """
        h, w = image.shape[:2]

        # ── PIL 预处理 ──────────────────────────────────────────────
        image_pil = Image.fromarray(image)
        image_pil = cvtColor(image_pil)
        image_data = resize_image(
            image_pil, (self.input_shape[1], self.input_shape[0]), self.letterbox
        )
        image_data = np.expand_dims(
            np.transpose(
                preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)
            ),
            0,
        )

        # ── 推理 ─────────────────────────────────────────────────────
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(
                torch.cat(outputs, 1),
                self.num_classes,
                self.input_shape,
                np.array(image_pil.size),
                self.letterbox,
                conf_thres=conf,
                nms_thres=nms_iou,
            )

        # ── 构建输出 ─────────────────────────────────────────────────
        annotated = image.copy()
        detections = []

        if results[0] is not None:
            top_label = np.array(results[0][:, 6], dtype="int32")
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]  # (y1, x1, y2, x2)

            for i, c in enumerate(top_label):
                predicted_class = self.class_names[int(c)]
                score = float(top_conf[i])
                y1, x1, y2, x2 = top_boxes[i]

                x1 = max(0, int(np.floor(x1)))
                y1 = max(0, int(np.floor(y1)))
                x2 = min(w, int(np.floor(x2)))
                y2 = min(h, int(np.floor(y2)))

                detections.append({
                    "label": predicted_class,
                    "confidence": round(score, 3),
                    "bbox": [x1, y1, x2, y2],
                })

                # 绘制检测框 (BGR 绘制再转回)
                color = self.colors[int(c) % len(self.colors)]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{predicted_class} {score:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1
                )
                cv2.putText(
                    annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

        return annotated, detections


if __name__ == "__main__":
    d = Detector()
    print(f"模型加载成功，可用类别 ({len(d.class_names)}):", d.class_names)
    print(f"使用设备: {'CUDA' if d.cuda else 'CPU'}")
