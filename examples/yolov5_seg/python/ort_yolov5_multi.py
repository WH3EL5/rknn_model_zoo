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
DEFAULT_ANCHOR_FILE = PROJECT_DIR / "model" / "anchors_yolov5.txt"


def load_default_anchors():
    # Fallback to standard YOLOv5 anchors when local anchor file is missing.
    fallback = np.array(
        [
            [[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]],
            [[30.0, 61.0], [62.0, 45.0], [59.0, 119.0]],
            [[116.0, 90.0], [156.0, 198.0], [373.0, 326.0]],
        ],
        dtype=np.float32,
    )

    if not DEFAULT_ANCHOR_FILE.exists():
        return fallback

    values = [float(v.strip()) for v in DEFAULT_ANCHOR_FILE.read_text().splitlines() if v.strip()]
    if len(values) != 18:
        return fallback

    return np.array(values, dtype=np.float32).reshape(3, 3, 2)


YOLOV5_ANCHORS = load_default_anchors()


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


def decode_yolov5_head(output: np.ndarray, anchors: np.ndarray, input_size):
    input_w, input_h = input_size
    _, _, grid_h, grid_w = output.shape
    na = anchors.shape[0]
    no = output.shape[1] // na

    feat = output.reshape(1, na, no, grid_h, grid_w)[0]

    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    grid = np.stack((col, row), axis=0).reshape(1, 2, grid_h, grid_w).astype(np.float32)
    stride = np.array([input_w / grid_w, input_h / grid_h], dtype=np.float32).reshape(1, 2, 1, 1)
    anchor_grid = anchors.astype(np.float32).reshape(na, 2, 1, 1)

    box_xy = feat[:, 0:2, :, :] * 2.0 - 0.5
    box_wh = (feat[:, 2:4, :, :] * 2.0) ** 2 * anchor_grid

    box_xy = (box_xy + grid) * stride
    box = np.concatenate((box_xy, box_wh), axis=1)
    box = box.transpose(0, 2, 3, 1).reshape(-1, 4)

    boxes_xyxy = xywh2xyxy(box)
    objectness = feat[:, 4:5, :, :].transpose(0, 2, 3, 1).reshape(-1)
    mask_dim = max(no - 5 - len(CLASSES), 0)
    class_dim = no - 5 - mask_dim

    class_scores = feat[:, 5 : 5 + class_dim, :, :].transpose(0, 2, 3, 1).reshape(-1, class_dim)
    if mask_dim > 0:
        mask_coeffs = feat[:, 5 + class_dim :, :, :].transpose(0, 2, 3, 1).reshape(-1, mask_dim)
    else:
        mask_coeffs = np.empty((boxes_xyxy.shape[0], 0), dtype=np.float32)

    return boxes_xyxy, objectness, class_scores, mask_coeffs


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def process_masks(proto: np.ndarray, mask_coeffs: np.ndarray, boxes: np.ndarray, input_size, threshold=0.5):
    input_w, input_h = input_size
    if proto.ndim == 4:
        proto = np.squeeze(proto, axis=0)
    if proto.ndim != 3:
        return None

    c, ph, pw = proto.shape
    if mask_coeffs.shape[1] != c:
        return None

    masks = sigmoid(mask_coeffs @ proto.reshape(c, -1)).reshape(-1, ph, pw)

    masks_out = np.zeros((masks.shape[0], input_h, input_w), dtype=np.uint8)
    scale_x = pw / float(input_w)
    scale_y = ph / float(input_h)

    for i in range(masks.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        x1p = int(np.clip(np.floor(x1 * scale_x), 0, pw - 1))
        y1p = int(np.clip(np.floor(y1 * scale_y), 0, ph - 1))
        x2p = int(np.clip(np.ceil(x2 * scale_x), 0, pw))
        y2p = int(np.clip(np.ceil(y2 * scale_y), 0, ph))
        if x2p <= x1p or y2p <= y1p:
            continue

        m = masks[i]
        crop = np.zeros_like(m, dtype=np.float32)
        crop[y1p:y2p, x1p:x2p] = m[y1p:y2p, x1p:x2p]
        m_full = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        masks_out[i] = (m_full > threshold).astype(np.uint8)

    return masks_out


def restore_masks_to_original(masks: np.ndarray, ratio: float, pad, original_shape):
    if masks is None or masks.size == 0:
        return None

    orig_h, orig_w = original_shape
    input_h, input_w = masks.shape[1:]
    new_w = int(round(orig_w * ratio))
    new_h = int(round(orig_h * ratio))
    left = int(round(pad[0] - 0.1))
    top = int(round(pad[1] - 0.1))

    x1 = max(0, left)
    y1 = max(0, top)
    x2 = min(input_w, left + new_w)
    y2 = min(input_h, top + new_h)

    restored = np.zeros((masks.shape[0], orig_h, orig_w), dtype=np.uint8)
    for i in range(masks.shape[0]):
        crop = masks[i, y1:y2, x1:x2]
        if crop.size == 0:
            continue
        restored[i] = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return restored


def post_process(outputs, input_size, obj_thresh, nms_thresh):
    if not outputs:
        return None, None, None

    det_heads = [o for o in outputs if isinstance(o, np.ndarray) and o.ndim == 4 and o.shape[1] % 3 == 0 and o.shape[1] >= 255]
    proto_candidates = [o for o in outputs if isinstance(o, np.ndarray) and o.ndim == 4 and o.shape[1] < 128]

    # Multi-output head mode: e.g. seg [(1,351,80,80), (1,351,40,40), (1,351,20,20), (1,32,160,160)].
    if len(det_heads) >= 3:
        outputs_sorted = sorted(det_heads[:3], key=lambda x: x.shape[-1], reverse=True)
        boxes_list, obj_list, cls_list, seg_list = [], [], [], []

        for i, out in enumerate(outputs_sorted):
            out = np.asarray(out, dtype=np.float32)
            if out.shape[0] != 1:
                raise ValueError(f"Unexpected batch size in YOLOv5 output: {out.shape}")

            boxes_i, obj_i, cls_i, seg_i = decode_yolov5_head(out, YOLOV5_ANCHORS[i], input_size)
            boxes_list.append(boxes_i)
            obj_list.append(obj_i)
            cls_list.append(cls_i)
            seg_list.append(seg_i)

        boxes = np.concatenate(boxes_list, axis=0)
        objectness = np.concatenate(obj_list, axis=0)
        class_scores = np.concatenate(cls_list, axis=0)
        seg_coeffs = np.concatenate(seg_list, axis=0) if seg_list else np.empty((boxes.shape[0], 0), dtype=np.float32)
        proto = proto_candidates[0] if proto_candidates else None
        classes = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(class_scores.shape[0]), classes]
        scores = objectness * class_conf
    else:
        # Single-output mode: usually [1, 25200, 85] or [1, 85, 25200].
        pred = np.squeeze(outputs[0])

        if pred.ndim != 2:
            raise ValueError(f"Unexpected YOLOv5 output shape: {outputs[0].shape}")

        if pred.shape[1] < pred.shape[0]:
            pred = pred.T

        if pred.shape[1] <= 5:
            raise ValueError(f"Invalid YOLOv5 prediction shape after transpose: {pred.shape}")

        boxes = xywh2xyxy(pred[:, :4])
        objectness = pred[:, 4]
        cls_dim = min(len(CLASSES), pred.shape[1] - 5)
        class_scores = pred[:, 5 : 5 + cls_dim]
        seg_coeffs = pred[:, 5 + cls_dim :] if pred.shape[1] > 5 + cls_dim else np.empty((pred.shape[0], 0), dtype=np.float32)
        proto = proto_candidates[0] if proto_candidates else None
        classes = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(class_scores.shape[0]), classes]
        scores = objectness * class_conf

    keep_mask = scores >= obj_thresh
    boxes = boxes[keep_mask]
    classes = classes[keep_mask]
    scores = scores[keep_mask]
    seg_coeffs = seg_coeffs[keep_mask] if seg_coeffs.size else seg_coeffs

    if boxes.size == 0:
        return None, None, None

    nboxes, nclasses, nscores, nseg = [], [], [], []
    for c in set(classes.tolist()):
        inds = np.where(classes == c)
        b = boxes[inds]
        cls = classes[inds]
        s = scores[inds]
        seg = seg_coeffs[inds] if seg_coeffs.size else None
        keep = nms_boxes(b, s, nms_thresh)

        if keep.size > 0:
            nboxes.append(b[keep])
            nclasses.append(cls[keep])
            nscores.append(s[keep])
            if seg is not None:
                nseg.append(seg[keep])

    if not nboxes:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    seg_coeffs = np.concatenate(nseg) if nseg else np.empty((boxes.shape[0], 0), dtype=np.float32)

    masks = None
    if proto is not None and seg_coeffs.size:
        masks = process_masks(proto=np.asarray(proto, dtype=np.float32), mask_coeffs=seg_coeffs, boxes=boxes, input_size=input_size)

    return boxes, classes, scores, masks


def draw_detections(image, boxes, scores, classes, masks=None):
    if masks is not None:
        color_bank = np.array(
            [
                [255, 56, 56], [255, 157, 151], [255, 112, 31], [255, 178, 29], [207, 210, 49],
                [72, 249, 10], [146, 204, 23], [61, 219, 134], [26, 147, 52], [0, 212, 187],
            ],
            dtype=np.uint8,
        )
        overlay = image.copy()
        for i, (cl, mask) in enumerate(zip(classes, masks)):
            color = color_bank[int(cl) % len(color_bank)].tolist()
            overlay[mask.astype(bool)] = color
        image[:] = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)

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


class YoloV5OrtDetector:
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

        outputs = self.session.run(None, {self.input_name: input_tensor})
        t0 = time.perf_counter()
        for i in range(10):
            outputs = self.session.run(None, {self.input_name: input_tensor})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 10.0

        boxes, classes, scores, masks = post_process(
            outputs,
            input_size=(self.input_w, self.input_h),
            obj_thresh=obj_thresh,
            nms_thresh=nms_thresh,
        )

        if boxes is None:
            return None, None, None, None, elapsed_ms

        masks = restore_masks_to_original(masks, ratio=ratio, pad=pad, original_shape=image_bgr.shape[:2])
        boxes = scale_boxes(boxes, ratio=ratio, pad=pad, original_shape=image_bgr.shape[:2])
        return boxes, classes, scores, masks, elapsed_ms


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


def main():
    parser = argparse.ArgumentParser("YOLOv5 ONNX Runtime inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="ONNX model path",
    )
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Image path or image directory")
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

    model_name = Path(model_path).stem
    print(f"\n=== Running model: {model_path} ===")

    detector = YoloV5OrtDetector(model_path, providers=providers, fallback_input_size=args.input_size)
    print(f"Input size: {detector.input_w}x{detector.input_h}")
    print(f"Providers: {detector.session.get_providers()}")

    times = []
    for img_path in images:
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        boxes, classes, scores, masks, elapsed_ms = detector.infer(
            image,
            obj_thresh=args.obj_thresh,
            nms_thresh=args.nms_thresh,
        )
        times.append(elapsed_ms)

        vis = image.copy()
        det_count = 0 if boxes is None else boxes.shape[0]
        if boxes is not None:
            draw_detections(vis, boxes, scores, classes, masks=masks)

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

# python python/ort_yolov5_multi.py --model model/yolov5s.onnx --source model/bus.jpg --save_dir python/result_ort --providers CPUExecutionProvider