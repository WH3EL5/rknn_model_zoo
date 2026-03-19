import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMG_SIZE = 448

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE = str(PROJECT_DIR / "model" / "picture.jpg")
DEFAULT_ENCODER = str(PROJECT_DIR / "model" / "mobilesam_encoder.onnx")
DEFAULT_DECODER = str(PROJECT_DIR / "model" / "mobilesam_decoder.onnx")
DEFAULT_COORDS = str(PROJECT_DIR / "model" / "coords.txt")
DEFAULT_LABELS = str(PROJECT_DIR / "model" / "labels.txt")

ENCODER_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
ENCODER_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def is_image_file(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def get_preprocess_shape(oldh, oldw, img_size):
    scale = float(img_size) / float(max(oldh, oldw))
    newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
    return newh, neww


def coords_preprocess(coords: np.ndarray, ori_shape, img_size):
    oldh, oldw = ori_shape
    newh, neww = get_preprocess_shape(oldh, oldw, img_size)
    out = coords.copy()
    out[..., 0] = out[..., 0] * (neww / oldw)
    out[..., 1] = out[..., 1] * (newh / oldh)
    return out


def resize_bilinear_chw(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    # arr: [N, C, H, W]
    n, c = arr.shape[:2]
    out = np.zeros((n, c, out_h, out_w), dtype=np.float32)
    for i in range(n):
        for j in range(c):
            out[i, j] = cv2.resize(arr[i, j], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return out


def postprocess(low_res_masks: np.ndarray, input_shape, ori_shape, img_size):
    # Match the original pipeline:
    # 1) low_res_masks -> (img_size, img_size)
    # 2) remove bottom/right padding
    # 3) resize back to original image size
    masks = resize_bilinear_chw(low_res_masks.astype(np.float32), img_size, img_size)
    in_h, in_w = input_shape
    masks = masks[:, :, :in_h, :in_w]
    masks = resize_bilinear_chw(masks, ori_shape[0], ori_shape[1])
    return masks


def parse_coords_arg(coords_arg: str) -> np.ndarray:
    p = Path(coords_arg)
    if p.exists():
        values = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) != 2:
                    raise ValueError(f"Each coords line must have 2 numbers, got: {line}")
                values.append([float(parts[0]), float(parts[1])])
        if not values:
            raise ValueError(f"No valid coords found in: {coords_arg}")
        return np.array(values, dtype=np.float32)

    pairs = []
    for item in coords_arg.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError("Invalid --point_coords format, expected like: '190,70;460,280'")
        pairs.append([float(parts[0]), float(parts[1])])

    if not pairs:
        raise ValueError("No valid point coords parsed")
    return np.array(pairs, dtype=np.float32)


def parse_labels_arg(labels_arg: str) -> np.ndarray:
    p = Path(labels_arg)
    if p.exists():
        values = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    values.append(float(line))
        if not values:
            raise ValueError(f"No valid labels found in: {labels_arg}")
        return np.array(values, dtype=np.float32)

    parts = [x.strip() for x in labels_arg.split(",") if x.strip()]
    if not parts:
        raise ValueError("No valid point labels parsed")
    return np.array([float(x) for x in parts], dtype=np.float32)


def draw_result(image, mask, point_coords, point_labels, color=(144, 144, 30)):
    vis = image.copy()

    alpha = 0.5
    color_arr = np.array(color, dtype=np.uint8)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_3c = np.stack([mask_u8, mask_u8, mask_u8], axis=-1)
    color_map = np.ones_like(mask_3c, dtype=np.uint8) * color_arr.reshape(1, 1, 3)
    blended = cv2.addWeighted(vis, alpha, color_map, 1.0 - alpha, 0)
    vis = np.where(mask_3c > 0, blended, vis)

    top_left = None
    bottom_right = None
    for coord, label in zip(point_coords.astype(int), point_labels.astype(int)):
        if label == 0:
            cv2.circle(vis, tuple(coord), 10, (0, 0, 255), 2)
        elif label == 1:
            cv2.circle(vis, tuple(coord), 10, (0, 255, 0), 2)
        elif label == 2:
            top_left = tuple(coord)
        elif label == 3:
            bottom_right = tuple(coord)

    if top_left is not None and bottom_right is not None:
        cv2.rectangle(vis, top_left, bottom_right, (0, 255, 0), 2)

    return vis


class MobileSamOrtPredictor:
    def __init__(self, encoder_path: str, decoder_path: str, providers, fallback_input_size=448):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path

        self.encoder_sess = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder_sess = ort.InferenceSession(decoder_path, providers=providers)

        self.encoder_input = self.encoder_sess.get_inputs()[0]
        self.encoder_input_name = self.encoder_input.name

        input_shape = self.encoder_input.shape
        self.input_size = int(fallback_input_size)
        if len(input_shape) == 4:
            dims = [d for d in input_shape if isinstance(d, int)]
            if len(dims) >= 3:
                cand = [d for d in dims if d != 1 and d != 3]
                if cand:
                    self.input_size = int(max(cand))

        self.decoder_input_names = [x.name for x in self.decoder_sess.get_inputs()]
        self.decoder_output_names = [x.name for x in self.decoder_sess.get_outputs()]

    def _prepare_encoder_input(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        oldh, oldw = image_rgb.shape[:2]
        newh, neww = get_preprocess_shape(oldh, oldw, self.input_size)

        resized = cv2.resize(image_rgb, (neww, newh), interpolation=cv2.INTER_LINEAR)
        padh, padw = self.input_size - newh, self.input_size - neww
        padded = cv2.copyMakeBorder(resized, 0, padh, 0, padw, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        arr = padded.astype(np.float32)
        arr = (arr - ENCODER_MEAN) / ENCODER_STD

        shape = self.encoder_input.shape
        if len(shape) != 4:
            raise ValueError(f"Unexpected encoder input shape: {shape}")

        if isinstance(shape[1], int) and shape[1] == 3:
            arr = arr.transpose(2, 0, 1)[None, ...]
        elif isinstance(shape[3], int) and shape[3] == 3:
            arr = arr[None, ...]
        else:
            # Dynamic shape case: default to NHWC used by this demo.
            arr = arr[None, ...]

        return arr, (newh, neww)

    def infer(self, image_bgr: np.ndarray, point_coords: np.ndarray, point_labels: np.ndarray, mask_input_path=None):
        encoder_input, input_shape = self._prepare_encoder_input(image_bgr)

        for _ in range(6):
            if _ == 1:
                t0 = time.perf_counter()
            image_embeddings = self.encoder_sess.run(None, {self.encoder_input_name: encoder_input})[0]
        encoder_ms = (time.perf_counter() - t0) * 1000.0 / 5

        coords = coords_preprocess(point_coords[None, :, :], image_bgr.shape[:2], self.input_size).astype(np.float32)
        labels = point_labels[None, :].astype(np.float32)

        if mask_input_path:
            mask_input = np.load(mask_input_path).astype(np.float32)
            has_mask_input = np.ones(1, dtype=np.float32)
        else:
            mask_input = np.zeros((1, 1, 112, 112), dtype=np.float32)
            has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_feed = {}
        input_name_set = set(self.decoder_input_names)

        if "image_embeddings" in input_name_set:
            decoder_feed["image_embeddings"] = image_embeddings
        if "point_coords" in input_name_set:
            decoder_feed["point_coords"] = coords
        if "point_labels" in input_name_set:
            decoder_feed["point_labels"] = labels
        if "mask_input" in input_name_set:
            decoder_feed["mask_input"] = mask_input
        if "has_mask_input" in input_name_set:
            decoder_feed["has_mask_input"] = has_mask_input
        if "orig_im_size" in input_name_set:
            decoder_feed["orig_im_size"] = np.array(image_bgr.shape[:2], dtype=np.float32)

        # Fallback for uncommon exported models with different names but same order.
        if len(decoder_feed) != len(self.decoder_input_names):
            ordered_inputs = [image_embeddings, coords, labels, mask_input, has_mask_input]
            if len(self.decoder_input_names) > 5:
                ordered_inputs.append(np.array(image_bgr.shape[:2], dtype=np.float32))
            decoder_feed = {
                name: ordered_inputs[idx]
                for idx, name in enumerate(self.decoder_input_names)
                if idx < len(ordered_inputs)
            }

        for _ in range(6):
            if _ == 1:
                t1 = time.perf_counter()
            decoder_outputs = self.decoder_sess.run(None, decoder_feed)
        decoder_ms = (time.perf_counter() - t1) * 1000.0 / 5

        output_map = {name: out for name, out in zip(self.decoder_output_names, decoder_outputs)}
        if "iou_predictions" in output_map:
            scores = output_map["iou_predictions"]
        else:
            scores = decoder_outputs[0]

        if "low_res_masks" in output_map:
            low_res_masks = output_map["low_res_masks"]
        else:
            low_res_masks = decoder_outputs[1]

        masks = postprocess(low_res_masks, input_shape=input_shape, ori_shape=image_bgr.shape[:2], img_size=self.input_size)
        return scores, masks, encoder_ms, decoder_ms


def main():
    parser = argparse.ArgumentParser("MobileSAM ONNX Runtime inference")
    parser.add_argument("--encoder", type=str, required=True, help="MobileSAM encoder ONNX model path")
    parser.add_argument("--decoder", type=str, required=True, help="MobileSAM decoder ONNX model path")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Image path or image directory")
    parser.add_argument("--point_coords", type=str, default=DEFAULT_COORDS, help="Point/box coords txt path or 'x1,y1;x2,y2'")
    parser.add_argument("--point_labels", type=str, default=DEFAULT_LABELS, help="Point labels txt path or '2,3'")
    parser.add_argument("--mask_input", type=str, default=None, help="Mask input .npy path, default uses zeros")
    parser.add_argument("--save_dir", type=str, default="./result_ort", help="Directory to save results")
    parser.add_argument("--mask_thresh", type=float, default=0.0, help="Mask threshold")
    parser.add_argument(
        "--providers",
        type=str,
        default="CPUExecutionProvider",
        help="Comma-separated ORT providers, e.g. CPUExecutionProvider or CUDAExecutionProvider,CPUExecutionProvider",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=IMG_SIZE,
        help="Fallback input size for dynamic ONNX input (used if model input shape is dynamic)",
    )
    parser.add_argument("--img_show", action="store_true", help="Show result windows")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    images = collect_images(args.source)
    os.makedirs(args.save_dir, exist_ok=True)

    encoder_path = resolve_model_path(args.encoder)
    decoder_path = resolve_model_path(args.decoder)

    point_coords = parse_coords_arg(args.point_coords)
    point_labels = parse_labels_arg(args.point_labels)
    if point_coords.shape[0] != point_labels.shape[0]:
        raise ValueError(
            f"point count mismatch: len(point_coords)={point_coords.shape[0]} vs len(point_labels)={point_labels.shape[0]}"
        )

    if args.mask_input is not None and not Path(args.mask_input).exists():
        raise FileNotFoundError(f"Mask input not found: {args.mask_input}")

    print(f"\n=== Running encoder model: {encoder_path} ===")
    print(f"=== Running decoder model: {decoder_path} ===")
    print(f"Points: {point_coords.tolist()}")
    print(f"Labels: {point_labels.astype(int).tolist()}")

    predictor = MobileSamOrtPredictor(
        encoder_path,
        decoder_path,
        providers=providers,
        fallback_input_size=args.input_size,
    )
    print(f"Input size: {predictor.input_size}x{predictor.input_size}")
    print(f"Encoder providers: {predictor.encoder_sess.get_providers()}")
    print(f"Decoder providers: {predictor.decoder_sess.get_providers()}")

    times = []
    for img_path in images:
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        scores, masks, encoder_ms, decoder_ms = predictor.infer(
            image,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input_path=args.mask_input,
        )

        best_idx = int(np.argmax(scores))
        best_score = float(scores.reshape(-1)[best_idx])
        best_mask = masks[:, best_idx, :, :] > args.mask_thresh
        vis = draw_result(image, best_mask[0], point_coords=point_coords, point_labels=point_labels)

        save_name = f"{Path(img_path).stem}_mobilesam_ort.jpg"
        save_path = str(Path(args.save_dir) / save_name)
        cv2.imwrite(save_path, vis)

        total_ms = encoder_ms + decoder_ms
        times.append(total_ms)
        print(
            f"{Path(img_path).name}: mask_score={best_score:.4f}, "
            f"encoder={encoder_ms:.2f} ms, decoder={decoder_ms:.2f} ms, total={total_ms:.2f} ms -> {save_path}"
        )

        if args.img_show:
            cv2.imshow(f"MobileSAM ORT - {Path(img_path).name}", vis)
            cv2.waitKey(0)

    if times:
        print(f"Average latency (MobileSAM ORT): {np.mean(times):.2f} ms over {len(times)} image(s)")

    if args.img_show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# python ort_mobilesam.py --encoder ../model/mobilesam_encoder.onnx --decoder ../model/mobilesam_decoder.onnx --source ../model/picture.jpg --save_dir ../python/result_ort --providers CPUExecutionProvider