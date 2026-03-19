import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CLASSES = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE = str(PROJECT_DIR / "model" / "bus.jpg")
DEFAULT_ANCHORS = str(PROJECT_DIR / "model" / "anchors_yolov7.txt")


def is_image_file(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(0, 0, 0)):
    """Resize image with unchanged aspect ratio using padding."""
    shape = im.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w, h

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


def clip_boxes(boxes: np.ndarray, shape):
    h, w = shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    return boxes


def scale_boxes(boxes: np.ndarray, ratio: float, pad, original_shape):
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    return clip_boxes(boxes, original_shape)


def box_process_yolov7(position: np.ndarray, anchors, input_size):
    input_w, input_h = input_size
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1).astype(np.float32)

    stride = np.array([input_w / grid_w, input_h / grid_h], dtype=np.float32).reshape(1, 2, 1, 1)

    anchors = np.array(anchors, dtype=np.float32).reshape(len(anchors), 2, 1, 1)

    box_xy = position[:, :2, :, :] * 2.0 - 0.5
    box_wh = (position[:, 2:4, :, :] * 2.0) ** 2 * anchors

    box_xy = (box_xy + grid) * stride
    xyxy = np.concatenate(
        (
            box_xy[:, 0:1, :, :] - box_wh[:, 0:1, :, :] / 2.0,
            box_xy[:, 1:2, :, :] - box_wh[:, 1:2, :, :] / 2.0,
            box_xy[:, 0:1, :, :] + box_wh[:, 0:1, :, :] / 2.0,
            box_xy[:, 1:2, :, :] + box_wh[:, 1:2, :, :] / 2.0,
        ),
        axis=1,
    )

    return xyxy


def xywh2xyxy(boxes_xywh: np.ndarray):
    boxes = boxes_xywh.copy()
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
    return boxes


def nms_boxes(boxes, scores, nms_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1e-5)
        h = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-12)

        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def post_process_yolov7(outputs, input_size, anchors, obj_thresh, nms_thresh):
    if not outputs:
        return None, None, None

    if len(outputs) == 3 and all(out.ndim == 4 for out in outputs):
        if len(anchors) != 3:
            raise ValueError(f"YOLOv7 expects 3 anchor groups, but got {len(anchors)}")

        boxes_all, cls_all, obj_all = [], [], []

        for i, out in enumerate(outputs):
            na = len(anchors[i])
            if out.shape[1] % na != 0:
                raise ValueError(
                    f"Output channel {out.shape[1]} is not divisible by anchor count {na} on scale {i}"
                )

            out = out.reshape(na, -1, out.shape[2], out.shape[3])
            boxes_all.append(box_process_yolov7(out[:, :4, :, :], anchors[i], input_size))
            obj_all.append(out[:, 4:5, :, :])
            cls_all.append(out[:, 5:, :, :])

        def flatten_per_level(x: np.ndarray):
            ch = x.shape[1]
            x = x.transpose(0, 2, 3, 1)
            return x.reshape(-1, ch)

        boxes = np.concatenate([flatten_per_level(v) for v in boxes_all])
        classes_conf = np.concatenate([flatten_per_level(v) for v in cls_all])
        objectness = np.concatenate([flatten_per_level(v) for v in obj_all]).reshape(-1)

        classes = np.argmax(classes_conf, axis=1)
        class_scores = classes_conf[np.arange(classes_conf.shape[0]), classes]
        scores = class_scores * objectness

        keep_mask = scores >= obj_thresh
        boxes = boxes[keep_mask]
        classes = classes[keep_mask]
        scores = scores[keep_mask]
    else:
        # Some exported YOLOv7 ONNX models provide decoded predictions [1, N, 85] / [N, 85].
        pred = outputs[0]
        pred = np.squeeze(pred)

        if pred.ndim != 2:
            raise ValueError(f"Unexpected YOLOv7 output shape: {outputs[0].shape}")

        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        if pred.shape[1] <= 5:
            raise ValueError(f"Invalid YOLOv7 prediction shape after transpose: {pred.shape}")

        boxes = xywh2xyxy(pred[:, :4])
        objectness = pred[:, 4]
        class_scores = pred[:, 5:]
        classes = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), classes] * objectness

        keep_mask = scores >= obj_thresh
        boxes = boxes[keep_mask]
        classes = classes[keep_mask]
        scores = scores[keep_mask]

    if boxes.size == 0:
        return None, None, None

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes.tolist()):
        inds = np.where(classes == c)
        b = boxes[inds]
        cls = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s, nms_thresh)

        if keep.size > 0:
            nboxes.append(b[keep])
            nclasses.append(cls[keep])
            nscores.append(s[keep])

    if not nboxes:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw_detections(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = CLASSES[int(cl)] if int(cl) < len(CLASSES) else str(int(cl))
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 180, 20), 2)
        cv2.putText(
            image,
            f"{label} {score:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )


class YoloV7OrtDetector:
    def __init__(self, model_path: str, providers, anchors, fallback_input_size=640):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.anchors = anchors

        input_shape = self.session.get_inputs()[0].shape
        # Typical shape: [1, 3, 640, 640]
        if len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            self.input_h = int(input_shape[2])
            self.input_w = int(input_shape[3])
        else:
            self.input_h = int(fallback_input_size)
            self.input_w = int(fallback_input_size)

    def infer(self, image_bgr: np.ndarray, obj_thresh=0.25, nms_thresh=0.45):
        padded, ratio, pad = letterbox(image_bgr, new_shape=(self.input_h, self.input_w), color=(0, 0, 0))
        image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        input_tensor = image_rgb.transpose(2, 0, 1).astype(np.float32)[None, ...] / 255.0

        outputs = self.session.run(None, {self.input_name: input_tensor})
        for i in range(6):
            if i == 1:
                t0 = time.perf_counter()
            outputs = self.session.run(None, {self.input_name: input_tensor})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 5

        boxes, classes, scores = post_process_yolov7(
            outputs,
            input_size=(self.input_w, self.input_h),
            anchors=self.anchors,
            obj_thresh=obj_thresh,
            nms_thresh=nms_thresh,
        )

        if boxes is None:
            return None, None, None, elapsed_ms

        boxes = scale_boxes(boxes, ratio=ratio, pad=pad, original_shape=image_bgr.shape[:2])
        return boxes, classes, scores, elapsed_ms


def collect_images(source: str):
    if os.path.isfile(source):
        if not is_image_file(source):
            raise ValueError(f"Input file is not a supported image: {source}")
        return [source]

    if os.path.isdir(source):
        images = [str(Path(source) / p) for p in sorted(os.listdir(source)) if is_image_file(p)]
        if not images:
            raise ValueError(f"No image files found in directory: {source}")
        return images

    raise FileNotFoundError(f"Source not found: {source}")


def resolve_model_path(model_arg: str) -> str:
    model_path = Path(model_arg)
    if model_path.exists():
        return str(model_path)

    raise FileNotFoundError(f"Model not found: {model_arg}")


def load_anchors(anchor_path: str):
    anchor_file = Path(anchor_path)
    if not anchor_file.exists():
        raise FileNotFoundError(f"Anchor file not found: {anchor_path}")

    values = []
    with anchor_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))

    if len(values) != 18:
        raise ValueError(f"YOLOv7 anchor file should contain 18 numbers, but got {len(values)}")

    return np.array(values, dtype=np.float32).reshape(3, 3, 2).tolist()


def main():
    parser = argparse.ArgumentParser("YOLOv7 ONNX Runtime inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="ONNX model path",
    )
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Image path or image directory")
    parser.add_argument("--anchors", type=str, default=DEFAULT_ANCHORS, help="YOLOv7 anchors txt path")
    parser.add_argument("--save_dir", type=str, default="./result_ort", help="Directory to save results")
    parser.add_argument("--obj_thresh", type=float, default=0.25, help="Objectness threshold")
    parser.add_argument("--nms_thresh", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--providers",
        type=str,
        default="CPUExecutionProvider",
        help="Comma-separated ORT providers, e.g. CPUExecutionProvider or CUDAExecutionProvider,CPUExecutionProvider",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=640,
        help="Fallback input size for dynamic ONNX input (used if model input shape is dynamic)",
    )
    parser.add_argument("--img_show", action="store_true", help="Show result windows")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    images = collect_images(args.source)
    os.makedirs(args.save_dir, exist_ok=True)

    model_path = resolve_model_path(args.model)
    anchors = load_anchors(args.anchors)

    model_name = Path(model_path).stem
    print(f"\n=== Running model: {model_path} ===")
    print(f"Anchors: {args.anchors}")

    detector = YoloV7OrtDetector(
        model_path,
        providers=providers,
        anchors=anchors,
        fallback_input_size=args.input_size,
    )
    print(f"Input size: {detector.input_w}x{detector.input_h}")
    print(f"Providers: {detector.session.get_providers()}")

    times = []
    for img_path in images:
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        boxes, classes, scores, elapsed_ms = detector.infer(
            image,
            obj_thresh=args.obj_thresh,
            nms_thresh=args.nms_thresh,
        )
        times.append(elapsed_ms)

        vis = image.copy()
        det_count = 0 if boxes is None else boxes.shape[0]
        if boxes is not None:
            draw_detections(vis, boxes, scores, classes)

        save_name = f"{Path(img_path).stem}_{model_name}.jpg"
        save_path = str(Path(args.save_dir) / save_name)
        cv2.imwrite(save_path, vis)

        print(f"{Path(img_path).name}: {det_count} objects, {elapsed_ms:.2f} ms -> {save_path}")

        if args.img_show:
            cv2.imshow(f"{model_name} - {Path(img_path).name}", vis)
            cv2.waitKey(0)

    if times:
        print(f"Average latency ({model_name}): {np.mean(times):.2f} ms over {len(times)} image(s)")

    if args.img_show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# python python/ort_yolov7_multi.py --model model/yolov7.onnx --source model/bus.jpg --save_dir python/result_ort --providers CPUExecutionProvider