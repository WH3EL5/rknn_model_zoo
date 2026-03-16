import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CLASSES = (
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_MODEL = str(PROJECT_DIR / "yolov8n-obb.onnx")
DEFAULT_SOURCE = str(PROJECT_DIR / "model" / "test.jpg")
DEFAULT_SAVE_DIR = str(SCRIPT_DIR / "result_ort")
DEFAULT_PROVIDERS = ["CPUExecutionProvider"]
DEFAULT_OBJ_THRESH = 0.5
DEFAULT_NMS_THRESH = 0.4
DEFAULT_INPUT_SIZE = 640
DEFAULT_IMG_SHOW = False


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


def scale_rboxes(rboxes: np.ndarray, ratio: float, pad, original_shape):
    if rboxes is None:
        return None

    h, w = original_shape
    scaled = rboxes.copy()
    scaled[:, 0] = (scaled[:, 0] - pad[0]) / ratio
    scaled[:, 1] = (scaled[:, 1] - pad[1]) / ratio
    scaled[:, 2] = scaled[:, 2] / ratio
    scaled[:, 3] = scaled[:, 3] / ratio
    scaled[:, 0] = scaled[:, 0].clip(0, w - 1)
    scaled[:, 1] = scaled[:, 1].clip(0, h - 1)
    scaled[:, 2] = scaled[:, 2].clip(1e-3, w)
    scaled[:, 3] = scaled[:, 3].clip(1e-3, h)
    return scaled


def sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def dfl(position: np.ndarray):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, axis=2)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = (y * acc).sum(axis=2)
    return y


def box_process(position: np.ndarray):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)

    position = dfl(position)
    return grid, position


def rotated_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    rect1 = ((float(box1[0]), float(box1[1])), (float(box1[2]), float(box1[3])), float(np.degrees(box1[4])))
    rect2 = ((float(box2[0]), float(box2[1])), (float(box2[2]), float(box2[3])), float(np.degrees(box2[4])))

    inter_type, inter_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    if inter_type == 0 or inter_pts is None:
        return 0.0

    inter_area = cv2.contourArea(inter_pts)
    union = float(box1[2] * box1[3] + box2[2] * box2[3] - inter_area)
    if union <= 1e-12:
        return 0.0
    return float(inter_area / union)


def nms_rotated(rboxes: np.ndarray, scores: np.ndarray, nms_thresh: float):
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ious = np.array([rotated_iou(rboxes[i], rboxes[j]) for j in order[1:]], dtype=np.float32)
        remain = np.where(ious <= nms_thresh)[0]
        order = order[remain + 1]

    return np.array(keep, dtype=np.int64)


def decode_obb_branch(feat: np.ndarray, stride: float, angle_flat: np.ndarray, angle_offset: int, obj_thresh: float):
    class_num = len(CLASSES)
    if feat.shape[1] < 64 + class_num:
        raise ValueError(f"Unexpected OBB branch channels: {feat.shape}")

    grid_h, grid_w = feat.shape[2], feat.shape[3]
    hw = grid_h * grid_w

    grid, pos = box_process(feat[:, :64, :, :])
    left_top = pos[:, 0:2, :, :]
    right_bottom = pos[:, 2:4, :, :]

    wh_add = left_top + right_bottom
    wh_sub = (right_bottom - left_top) / 2.0

    ang = angle_flat[angle_offset: angle_offset + hw]
    if ang.shape[0] != hw:
        raise ValueError("Angle feature size does not match feature map size")
    ang = (ang - 0.25) * np.pi
    ang = ang.reshape(1, 1, grid_h, grid_w)

    cosv = np.cos(ang)
    sinv = np.sin(ang)
    x_rot = wh_sub[:, 0:1, :, :] * cosv - wh_sub[:, 1:2, :, :] * sinv
    y_rot = wh_sub[:, 0:1, :, :] * sinv + wh_sub[:, 1:2, :, :] * cosv

    cx = (x_rot + grid[:, 0:1, :, :] + 0.5) * stride
    cy = (y_rot + grid[:, 1:2, :, :] + 0.5) * stride
    ww = wh_add[:, 0:1, :, :] * stride
    hh = wh_add[:, 1:2, :, :] * stride

    boxes = np.concatenate((cx, cy, ww, hh, ang), axis=1)
    boxes = boxes.transpose(0, 2, 3, 1).reshape(-1, 5).astype(np.float32)

    cls_scores = sigmoid(feat[:, 64:64 + class_num, :, :])
    cls_scores = cls_scores.transpose(0, 2, 3, 1).reshape(-1, class_num)

    rows, cls_ids = np.where(cls_scores >= obj_thresh)
    if rows.size == 0:
        return None, None, None

    scores = cls_scores[rows, cls_ids].astype(np.float32)
    return boxes[rows], cls_ids.astype(np.int64), scores


def decode_obb_single_output(pred: np.ndarray, obj_thresh: float):
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ValueError(f"Unexpected OBB output shape: {pred.shape}")

    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    if pred.shape[1] < 6:
        raise ValueError(f"Invalid OBB prediction shape: {pred.shape}")

    class_num = len(CLASSES)
    if pred.shape[1] < 5 + class_num:
        class_num = pred.shape[1] - 5

    boxes = pred[:, :4].astype(np.float32)
    cls_scores = pred[:, 4:4 + class_num].astype(np.float32)
    angles = pred[:, 4 + class_num].astype(np.float32)

    if np.max(cls_scores) > 1.0 or np.min(cls_scores) < 0.0:
        cls_scores = sigmoid(cls_scores)

    if np.min(angles) >= 0.0 and np.max(angles) <= 1.0:
        angles = (angles - 0.25) * np.pi

    rows, cls_ids = np.where(cls_scores >= obj_thresh)
    if rows.size == 0:
        return None, None, None

    scores = cls_scores[rows, cls_ids].astype(np.float32)
    selected_boxes = boxes[rows]
    selected_angles = angles[rows]
    rboxes = np.concatenate([selected_boxes, selected_angles[:, None]], axis=1)

    return rboxes, cls_ids.astype(np.int64), scores


def post_process(outputs, input_size, obj_thresh, nms_thresh):
    if not outputs:
        return None, None, None

    model_w, model_h = input_size
    feat_maps = [o for o in outputs if o.ndim == 4 and o.shape[1] >= 64 + len(CLASSES)]

    if len(feat_maps) >= 3 and len(outputs) >= 4:
        total_cells = sum(int(f.shape[2] * f.shape[3]) for f in feat_maps)
        angle_candidates = [o for o in outputs if np.prod(o.shape[1:]) == total_cells]
        if not angle_candidates:
            raise ValueError("Unable to find OBB angle output tensor")

        angle_flat = angle_candidates[0].reshape(-1).astype(np.float32)
        all_boxes, all_classes, all_scores = [], [], []
        offset = 0

        for feat in feat_maps:
            stride = float(model_h / feat.shape[2])
            b, c, s = decode_obb_branch(feat.astype(np.float32), stride, angle_flat, offset, obj_thresh)
            offset += int(feat.shape[2] * feat.shape[3])
            if b is None:
                continue
            all_boxes.append(b)
            all_classes.append(c)
            all_scores.append(s)

        if not all_boxes:
            return None, None, None

        rboxes = np.concatenate(all_boxes)
        classes = np.concatenate(all_classes)
        scores = np.concatenate(all_scores)
    else:
        rboxes, classes, scores = decode_obb_single_output(outputs[0], obj_thresh)

    if rboxes is None or rboxes.size == 0:
        return None, None, None

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes.tolist()):
        inds = np.where(classes == c)[0]
        b = rboxes[inds]
        cls = classes[inds]
        s = scores[inds]
        keep = nms_rotated(b, s, nms_thresh)

        if keep.size > 0:
            nboxes.append(b[keep])
            nclasses.append(cls[keep])
            nscores.append(s[keep])

    if not nboxes:
        return None, None, None

    rboxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return rboxes, classes, scores


def draw_detections(image, rboxes, scores, classes):
    for box, score, cl in zip(rboxes, scores, classes):
        cx, cy, bw, bh, ang = [float(v) for v in box]
        label = CLASSES[int(cl)] if int(cl) < len(CLASSES) else str(int(cl))

        rect = ((cx, cy), (bw, bh), float(np.degrees(ang)))
        poly = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(image, [poly], True, (40, 180, 20), 1, lineType=cv2.LINE_AA)

        p0 = poly[np.argmin(poly[:, 1] + poly[:, 0])]
        cv2.putText(
            image,
            f"{label} {score:.2f}",
            (int(p0[0]), max(int(p0[1]) - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1,
        )


class YoloV8OrtDetector:
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
        padded, ratio, pad = letterbox(image_bgr, new_shape=(self.input_h, self.input_w), color=(114, 114, 114))
        image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        input_tensor = image_rgb.transpose(2, 0, 1).astype(np.float32)[None, ...] / 255.0

        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rboxes, classes, scores = post_process(
            outputs,
            input_size=(self.input_w, self.input_h),
            obj_thresh=obj_thresh,
            nms_thresh=nms_thresh,
        )

        if rboxes is None:
            return None, None, None, elapsed_ms

        rboxes = scale_rboxes(rboxes, ratio=ratio, pad=pad, original_shape=image_bgr.shape[:2])
        return rboxes, classes, scores, elapsed_ms


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
    model_path = resolve_model_path(DEFAULT_MODEL)
    images = collect_images(DEFAULT_SOURCE)
    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)

    model_name = Path(model_path).stem
    print(f"\n=== Running model: {model_path} ===")

    detector = YoloV8OrtDetector(model_path, providers=DEFAULT_PROVIDERS, fallback_input_size=DEFAULT_INPUT_SIZE)
    print(f"Input size: {detector.input_w}x{detector.input_h}")
    print(f"Providers: {detector.session.get_providers()}")

    times = []
    for img_path in images:
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        rboxes, classes, scores, elapsed_ms = detector.infer(
            image,
            obj_thresh=DEFAULT_OBJ_THRESH,
            nms_thresh=DEFAULT_NMS_THRESH,
        )
        times.append(elapsed_ms)

        vis = image.copy()
        det_count = 0 if rboxes is None else rboxes.shape[0]
        if rboxes is not None:
            draw_detections(vis, rboxes, scores, classes)

        save_name = f"{Path(img_path).stem}_{model_name}.jpg"
        save_path = str(Path(DEFAULT_SAVE_DIR) / save_name)
        cv2.imwrite(save_path, vis)
        legacy_result_path = str(SCRIPT_DIR / "result.jpg")
        cv2.imwrite(legacy_result_path, vis)

        print(f"{Path(img_path).name}: {det_count} objects, {elapsed_ms:.2f} ms -> {save_path}")

        if DEFAULT_IMG_SHOW:
            cv2.imshow(f"{model_name} - {Path(img_path).name}", vis)
            cv2.waitKey(0)

    if times:
        print(f"Average latency ({model_name}): {np.mean(times):.2f} ms over {len(times)} image(s)")

    if DEFAULT_IMG_SHOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()