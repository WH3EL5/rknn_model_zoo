import argparse
import os
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


def is_image_file(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im


def parse_input_hw(model_path: str):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    input_name = inp.name
    input_shape = inp.shape

    if not (len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int)):
        raise ValueError(f"Model input shape must be static NCHW, got: {input_shape}")

    input_h, input_w = int(input_shape[2]), int(input_shape[3])

    return input_name, input_h, input_w


def collect_images(calib_source: str, max_images: int) -> List[str]:
    src = Path(calib_source)
    if not src.exists():
        raise FileNotFoundError(f"Calibration source not found: {calib_source}")

    if src.is_file():
        if src.suffix.lower() == ".txt":
            images: List[str] = []
            for line in src.read_text().splitlines():
                p = line.strip()
                if not p:
                    continue
                image_path = Path(p)
                if not image_path.is_absolute():
                    image_path = (src.parent / image_path).resolve()
                if image_path.exists() and is_image_file(str(image_path)):
                    images.append(str(image_path))
        else:
            if not is_image_file(str(src)):
                raise ValueError(f"Calibration file is not an image: {calib_source}")
            images = [str(src.resolve())]
    else:
        images = [str((src / p).resolve()) for p in sorted(os.listdir(src)) if is_image_file(p)]

    if not images:
        raise ValueError(f"No calibration images found in: {calib_source}")

    return images[:max_images]


def preprocess_image(image_path: str, input_h: int, input_w: int) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read calibration image: {image_path}")

    padded = letterbox(image, new_shape=(input_h, input_w), color=(0, 0, 0))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32)[None, ...] / 255.0
    return tensor


class YoloV6CalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_path: str, image_paths: Iterable[str]):
        self.input_name, self.input_h, self.input_w = parse_input_hw(model_path)
        self.image_paths = list(image_paths)
        self.iterator = iter(self.image_paths)

    def get_next(self):
        try:
            image_path = next(self.iterator)
        except StopIteration:
            return None

        data = preprocess_image(image_path, self.input_h, self.input_w)
        return {self.input_name: data}

    def rewind(self):
        self.iterator = iter(self.image_paths)


def find_models(model_dir: str, pattern: str) -> List[str]:
    base = Path(model_dir)
    if not base.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    models = sorted(str(p.resolve()) for p in base.glob(pattern) if p.is_file())
    if not models:
        raise ValueError(f"No ONNX models found in {model_dir} with pattern: {pattern}")

    return models


def quantize_one_model(
    model_path: str,
    output_path: str,
    image_paths: List[str],
    calibrate_method: CalibrationMethod,
    activation_type: QuantType,
    weight_type: QuantType,
    per_channel: bool,
    reduce_range: bool,
):
    print(f"\n[INFO] Quantizing: {model_path}")
    print(f"[INFO] Output: {output_path}")

    data_reader = YoloV6CalibrationDataReader(model_path, image_paths)

    # QDQ + static calibration usually gives higher quantized-op coverage while keeping graph IO unchanged.
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=data_reader,
        calibrate_method=calibrate_method,
        quant_format=QuantFormat.QDQ,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=per_channel,
        reduce_range=reduce_range,
    )


def build_output_name(input_model: str, suffix: str) -> str:
    p = Path(input_model)
    return str(p.with_name(f"{p.stem}{suffix}.onnx"))


def parse_args():
    parser = argparse.ArgumentParser("Batch quantize YOLOv6 ONNX models with ONNX Runtime")
    parser.add_argument("--model_dir", type=str, default="../model", help="Directory containing ONNX models")
    parser.add_argument("--pattern", type=str, default="yolov6*.onnx", help="Glob pattern for model discovery")
    parser.add_argument(
        "--calib_source",
        type=str,
        default="../model",
        help="Calibration images source: image, directory, or txt list",
    )
    parser.add_argument("--max_calib", type=int, default=64, help="Max number of calibration images")
    parser.add_argument("--suffix", type=str, default="_int8", help="Output model suffix")
    parser.add_argument(
        "--calib_method",
        type=str,
        default="entropy",
        choices=["minmax", "entropy", "percentile"],
        help="Calibration method for static quantization",
    )
    parser.add_argument(
        "--activation_type",
        type=str,
        default="qint8",
        choices=["qint8", "quint8"],
        help="Activation quant type",
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        default="qint8",
        choices=["qint8", "quint8"],
        help="Weight quant type",
    )
    parser.add_argument("--reduce_range", action="store_true", help="Use reduced quantization range")

    return parser.parse_args()


def to_calib_method(name: str) -> CalibrationMethod:
    mapping = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    return mapping[name]


def to_quant_type(name: str) -> QuantType:
    return QuantType.QInt8 if name == "qint8" else QuantType.QUInt8


def main():
    args = parse_args()

    all_models = find_models(args.model_dir, args.pattern)
    models = [m for m in all_models if not Path(m).stem.endswith(args.suffix)]
    if not models:
        raise ValueError(
            f"No source models to quantize. All matched models already end with suffix: {args.suffix}"
        )

    image_paths = collect_images(args.calib_source, args.max_calib)

    print(f"[INFO] Found {len(models)} model(s)")
    for m in models:
        print(f"  - {m}")

    print(f"[INFO] Calibration images: {len(image_paths)}")

    calib_method = to_calib_method(args.calib_method)
    activation_type = to_quant_type(args.activation_type)
    weight_type = to_quant_type(args.weight_type)
    per_channel = False

    for model_path in models:
        out_path = build_output_name(model_path, args.suffix)
        quantize_one_model(
            model_path=model_path,
            output_path=out_path,
            image_paths=image_paths,
            calibrate_method=calib_method,
            activation_type=activation_type,
            weight_type=weight_type,
            per_channel=per_channel,
            reduce_range=args.reduce_range,
        )

    print("\n[INFO] Done. Use quantized models by changing only --models paths in your existing inference script.")


if __name__ == "__main__":
    main()
