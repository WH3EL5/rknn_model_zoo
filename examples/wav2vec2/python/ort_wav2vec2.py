import argparse
import os
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort


SAMPLE_RATE = 16000
DEFAULT_SOURCE = "../model/test.wav"
DEFAULT_MODEL = "../model/wav2vec2_base_960h_20s.onnx"

TOKENIZER_DICT = {
	0: "<pad>",
	1: "<s>",
	2: "</s>",
	3: "<unk>",
	4: "|",
	5: "E",
	6: "T",
	7: "A",
	8: "O",
	9: "N",
	10: "I",
	11: "H",
	12: "S",
	13: "R",
	14: "D",
	15: "L",
	16: "U",
	17: "M",
	18: "W",
	19: "C",
	20: "F",
	21: "G",
	22: "Y",
	23: "P",
	24: "B",
	25: "V",
	26: "K",
	27: "'",
	28: "X",
	29: "J",
	30: "Q",
	31: "Z",
}


def ensure_sample_rate(waveform, original_sample_rate, desired_sample_rate=SAMPLE_RATE):
	if original_sample_rate != desired_sample_rate:
		if waveform.size == 0:
			return waveform.astype(np.float32), desired_sample_rate
		desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
		old_positions = np.arange(len(waveform), dtype=np.float32)
		new_positions = np.linspace(0, len(waveform) - 1, desired_length, dtype=np.float32)
		waveform = np.interp(new_positions, old_positions, waveform).astype(np.float32)
	return waveform, desired_sample_rate


def ensure_channels(waveform, desired_channels=1):
	if waveform.ndim == 1:
		return waveform
	if desired_channels != 1:
		raise ValueError("Only mono audio is supported in this demo")
	return np.mean(waveform, axis=1)


def read_wave_mono(audio_path):
	with wave.open(audio_path, "rb") as wf:
		channels = wf.getnchannels()
		sample_rate = wf.getframerate()
		sample_width = wf.getsampwidth()
		frame_count = wf.getnframes()
		raw = wf.readframes(frame_count)

	if sample_width == 1:
		data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
		data = (data - 128.0) / 128.0
	elif sample_width == 2:
		data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
	elif sample_width == 4:
		data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
	else:
		raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

	if channels > 1:
		data = data.reshape(-1, channels)
	data = ensure_channels(data)
	return data.astype(np.float32), sample_rate


def pad_or_trim(audio_array, max_length, pad_value=0.0):
	array_length = len(audio_array)
	if array_length < max_length:
		pad_length = max_length - array_length
		return np.pad(audio_array, (0, pad_length), mode="constant", constant_values=pad_value)
	if array_length > max_length:
		return audio_array[:max_length]
	return audio_array


def compress_sequence(sequence):
	if len(sequence) == 0:
		return []

	compressed = [sequence[0]]
	for i in range(1, len(sequence)):
		if sequence[i] != sequence[i - 1]:
			compressed.append(sequence[i])
	return compressed


def decode(token_ids):
	token_ids = compress_sequence(token_ids)
	transcriptions = []
	for token_id in token_ids:
		if token_id <= 4:
			if token_id == 4:
				transcriptions.append(" ")
			continue
		transcriptions.append(TOKENIZER_DICT.get(int(token_id), ""))
	return "".join(transcriptions).strip()


def post_process(output):
	predicted_ids = np.argmax(output, axis=-1)
	return decode(predicted_ids[0].tolist())


def parse_providers(provider_string):
	providers = [p.strip() for p in provider_string.split(",") if p.strip()]
	if not providers:
		raise ValueError("--providers cannot be empty")
	return providers


def build_session(model_path, providers):
	available = ort.get_available_providers()
	valid = [p for p in providers if p in available]
	if not valid:
		raise ValueError(f"No valid providers found. Requested={providers}, available={available}")
	return ort.InferenceSession(model_path, providers=valid)


def get_input_samples(session, fallback):
	shape = session.get_inputs()[0].shape
	if len(shape) != 2:
		return fallback
	width = shape[1]
	if isinstance(width, int) and width > 0:
		return width
	return fallback


def infer_one(audio_path, session, input_samples):
	audio_data, sample_rate = read_wave_mono(audio_path)
	audio_data, _ = ensure_sample_rate(audio_data, sample_rate)
	audio_array = np.asarray(audio_data, dtype=np.float32)
	audio_array = pad_or_trim(audio_array, input_samples)
	audio_array = np.expand_dims(audio_array, axis=0)

	input_name = session.get_inputs()[0].name
	outputs = session.run(None, {input_name: audio_array})[0]
	return post_process(outputs)


def collect_audio_files(source):
	source_path = Path(source)
	if source_path.is_file():
		return [source_path]

	if not source_path.is_dir():
		raise FileNotFoundError(f"Source not found: {source}")

	files = [p for p in source_path.rglob("*") if p.suffix.lower() == ".wav"]
	if not files:
		raise FileNotFoundError(f"No wav files found in directory: {source}")
	return sorted(files)


def main():
	parser = argparse.ArgumentParser("Wav2vec2 ONNX Runtime inference")
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Wav2vec2 ONNX model path")
	parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Wav path or wav directory")
	parser.add_argument("--save_dir", type=str, default="./result_ort", help="Directory to save transcript results")
	parser.add_argument(
		"--providers",
		type=str,
		default="CPUExecutionProvider",
		help="Comma-separated ORT providers, e.g. CPUExecutionProvider or CUDAExecutionProvider,CPUExecutionProvider",
	)
	parser.add_argument(
		"--input_size",
		type=int,
		default=320000,
		help="Fallback number of audio samples for dynamic model input shape",
	)
	args = parser.parse_args()

	providers = parse_providers(args.providers)
	session = build_session(args.model, providers)
	input_samples = get_input_samples(session, args.input_size)

	audio_files = collect_audio_files(args.source)
	os.makedirs(args.save_dir, exist_ok=True)
	output_path = Path(args.save_dir) / "transcripts.txt"

	print(f"=== Running model: {args.model} ===")
	print(f"Input samples: {input_samples}")
	print(f"Providers: {session.get_providers()}")

	lines = []
	for audio_file in audio_files:
		for i in range(6):
			if i == 1:
				start = time.perf_counter()
			text = infer_one(str(audio_file), session, input_samples)
		elapsed_ms = (time.perf_counter() - start) * 1000.0 / 5

		line = f"{audio_file}\t{text}"
		lines.append(line)

		per_file_out = Path(args.save_dir) / f"{audio_file.stem}.txt"
		with open(per_file_out, "w", encoding="utf-8") as f:
			f.write(text + "\n")

		print(f"{audio_file.name}: 1 transcript -> {per_file_out}")
		with wave.open(str(audio_file), "rb") as wf:
			audio_sec = wf.getnframes() / float(wf.getframerate())
		audio_sec = max(audio_sec, 1e-9)
		print(f"    Transcript: {text}")
		print(f"    Average latency (wav2vec2): {elapsed_ms:.2f} ms for a {audio_sec:.2f}s audio")
		print(f"    RTF: {elapsed_ms / (audio_sec * 1000.0):.4f}")

if __name__ == "__main__":
	main()


'''
python ort_wav2vec2.py \
  --model ../model/wav2vec2_base_960h_20s.onnx \
  --source ../model/test.wav \
  --providers CPUExecutionProvider \
  --save_dir ./result_ort
'''
