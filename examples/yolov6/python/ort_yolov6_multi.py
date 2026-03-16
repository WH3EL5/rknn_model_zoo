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


def softmax(x: np.ndarray, axis: int):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def dfl(position: np.ndarray):
    """Distribution Focal Loss decode implemented with numpy."""
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, axis=2)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = (y * acc).sum(axis=2)
    return y


def box_process(position: np.ndarray, input_size):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)

    stride = np.array([input_size[1] / grid_h, input_size[0] / grid_w], dtype=np.float32).reshape(1, 2, 1, 1)

    if position.shape[1] == 4:
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    else:
        position = dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def filter_boxes(boxes, box_confidences, box_class_probs, obj_thresh):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    valid = np.where(class_max_score * box_confidences >= obj_thresh)
    scores = (class_max_score * box_confidences)[valid]
    boxes = boxes[valid]
    classes = classes[valid]

    return boxes, classes, scores


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


def post_process(outputs, input_size, obj_thresh, nms_thresh):
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(outputs) // default_branch

    if pair_per_branch < 2:
        raise ValueError(
            f"Unexpected output count: {len(outputs)}. Need at least 6 outputs for YOLOv6 decoupled head."
        )

    for i in range(default_branch):
        boxes.append(box_process(outputs[pair_per_branch * i], input_size))
        classes_conf.append(outputs[pair_per_branch * i + 1])
        scores.append(np.ones_like(outputs[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def flatten_per_level(x):
        ch = x.shape[1]
        x = x.transpose(0, 2, 3, 1)
        return x.reshape(-1, ch)

    boxes = np.concatenate([flatten_per_level(v) for v in boxes])
    classes_conf = np.concatenate([flatten_per_level(v) for v in classes_conf])
    scores = np.concatenate([flatten_per_level(v) for v in scores])

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf, obj_thresh)
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


class YoloV6OrtDetector:
    def __init__(self, model_path: str, providers, fallback_input_size=640):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

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

        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        boxes, classes, scores = post_process(
            outputs,
            input_size=(self.input_w, self.input_h),
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


def main():
    parser = argparse.ArgumentParser("YOLOv6 ONNX Runtime multi-model inference")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["../model/yolov6n.onnx", "../model/yolov6s.onnx", "../model/yolov6m.onnx"],
        help="One or more ONNX model paths",
    )
    parser.add_argument("--source", type=str, default="../model/bus.jpg", help="Image path or image directory")
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

    for model_path in args.models:
        if not os.path.exists(model_path):
            print(f"[WARN] Model not found, skip: {model_path}")
            continue

        model_name = Path(model_path).stem
        print(f"\n=== Running model: {model_path} ===")

        detector = YoloV6OrtDetector(model_path, providers=providers, fallback_input_size=args.input_size)
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
