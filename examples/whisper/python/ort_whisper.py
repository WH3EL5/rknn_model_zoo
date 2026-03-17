import argparse
import os
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
DEFAULT_SOURCE = "../model/test_en.wav"
DEFAULT_ENCODER = "../model/whisper_encoder_base_20s.onnx"
DEFAULT_DECODER = "../model/whisper_decoder_base_20s.onnx"
DEFAULT_VOCAB_EN = "../model/vocab_en.txt"
DEFAULT_VOCAB_ZH = "../model/vocab_zh.txt"
DEFAULT_MEL_FILTER = "../model/mel_80_filters.txt"


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


def get_char_index(c):
    if "A" <= c <= "Z":
        return ord(c) - ord("A")
    if "a" <= c <= "z":
        return ord(c) - ord("a") + (ord("Z") - ord("A") + 1)
    if "0" <= c <= "9":
        return ord(c) - ord("0") + (ord("Z") - ord("A")) + (ord("z") - ord("a")) + 2
    if c == "+":
        return 62
    if c == "/":
        return 63
    raise ValueError(f"Unknown base64 char: {c}")


def base64_decode(encoded_string):
    if not encoded_string:
        return ""

    output_length = len(encoded_string) // 4 * 3
    decoded_string = bytearray(output_length)

    index = 0
    output_index = 0
    while index < len(encoded_string):
        if encoded_string[index] == "=":
            break

        first_byte = (get_char_index(encoded_string[index]) << 2) + (
            (get_char_index(encoded_string[index + 1]) & 0x30) >> 4
        )
        decoded_string[output_index] = first_byte

        if index + 2 < len(encoded_string) and encoded_string[index + 2] != "=":
            second_byte = ((get_char_index(encoded_string[index + 1]) & 0x0F) << 4) + (
                (get_char_index(encoded_string[index + 2]) & 0x3C) >> 2
            )
            decoded_string[output_index + 1] = second_byte

            if index + 3 < len(encoded_string) and encoded_string[index + 3] != "=":
                third_byte = ((get_char_index(encoded_string[index + 2]) & 0x03) << 6) + get_char_index(
                    encoded_string[index + 3]
                )
                decoded_string[output_index + 2] = third_byte
                output_index += 3
            else:
                output_index += 2
        else:
            output_index += 1
        index += 4

    return decoded_string.decode("utf-8", errors="replace")


def read_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                key = parts[0]
                value = ""
            else:
                key, value = parts[0], parts[1]
            vocab[key] = value
    return vocab


def mel_filters(filters_path, n_mels):
    if n_mels != 80:
        raise ValueError(f"Unsupported n_mels: {n_mels}, only 80 is supported by provided filter file")
    mels_data = np.loadtxt(filters_path, dtype=np.float32).reshape((80, 201))
    return mels_data


def log_mel_spectrogram(audio, n_mels, filters_path):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        audio = audio.reshape(-1)

    if audio.size < N_FFT:
        audio = np.pad(audio, (0, N_FFT - audio.size))

    window = np.hanning(N_FFT).astype(np.float32)
    frame_count = max(1, 1 + (audio.size - N_FFT) // HOP_LENGTH)
    magnitudes = np.empty((N_FFT // 2 + 1, frame_count), dtype=np.float32)

    for i in range(frame_count):
        start = i * HOP_LENGTH
        frame = audio[start : start + N_FFT]
        if frame.size < N_FFT:
            frame = np.pad(frame, (0, N_FFT - frame.size))
        frame = frame * window
        spec = np.fft.rfft(frame, n=N_FFT)
        magnitudes[:, i] = (np.abs(spec) ** 2).astype(np.float32)

    magnitudes = magnitudes[:, :-1]
    filters = mel_filters(filters_path, n_mels)
    mel_spec = np.matmul(filters, magnitudes)

    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def pad_or_trim(audio_array, max_length):
    x_mel = np.zeros((N_MELS, max_length), dtype=np.float32)
    real_length = min(audio_array.shape[1], max_length)
    x_mel[:, :real_length] = audio_array[:, :real_length]
    return x_mel


def parse_providers(provider_string):
    providers = [p.strip() for p in provider_string.split(",") if p.strip()]
    if not providers:
        raise ValueError("--providers cannot be empty")
    return providers


def build_session(model_path, providers):
    available = ort.get_available_providers()
    valid = [p for p in providers if p in available]
    if not valid:
        raise ValueError(
            f"No valid providers found. Requested={providers}, available={available}"
        )
    return ort.InferenceSession(model_path, providers=valid)


def get_encoder_width(encoder_session, fallback):
    shape = encoder_session.get_inputs()[0].shape
    if len(shape) != 3:
        return fallback
    width = shape[2]
    if isinstance(width, int) and width > 0:
        return width
    return fallback


def get_decoder_token_len(decoder_session, fallback):
    token_shape = decoder_session.get_inputs()[0].shape
    if len(token_shape) != 2:
        return fallback
    token_len = token_shape[1]
    if isinstance(token_len, int) and token_len > 0:
        return token_len
    return fallback


def load_audio_to_mel(audio_path, input_size, n_mels, mel_filters_path):
    audio_data, sample_rate = read_wave_mono(audio_path)
    audio_data, _ = ensure_sample_rate(audio_data, sample_rate)

    audio_array = np.array(audio_data, dtype=np.float32)
    mel = log_mel_spectrogram(audio_array, n_mels, mel_filters_path)
    x_mel = pad_or_trim(mel, input_size)
    return np.expand_dims(x_mel, 0)


def run_decoder(decoder_session, encoder_out, vocab, task_code, decoder_input_len, max_steps, show_tokens=False):
    end_token = 50257
    timestamp_begin = 50364
    tokens = [50258, task_code, 50359, 50363]

    # Keep behavior aligned with project demo where decoder input is fixed length.
    base_len = max(4, decoder_input_len)
    if len(tokens) < base_len:
        repeat = int(np.ceil(base_len / len(tokens)))
        tokens = (tokens * repeat)[:base_len]
    else:
        tokens = tokens[:base_len]

    text_tokens = []
    pop_id = len(tokens)
    steps = 0

    while steps < max_steps:
        out_decoder = decoder_session.run(
            None,
            {
                "tokens": np.asarray([tokens], dtype=np.int64),
                "audio": encoder_out,
            },
        )[0]
        next_token = int(np.argmax(out_decoder[0, -1]))
        if next_token == end_token:
            break

        if next_token > timestamp_begin:
            steps += 1
            continue

        token_str = vocab.get(str(next_token), "")
        text_tokens.append(token_str)

        tokens.append(next_token)
        if pop_id > 4:
            pop_id -= 1
        if pop_id < len(tokens):
            tokens.pop(pop_id)

        if show_tokens:
            print(f"token={next_token}, piece={token_str}")
        steps += 1

    result = "".join(text_tokens).replace("\u0120", " ").replace("<|endoftext|>", "").replace("\n", "")
    if task_code == 50260:
        result = base64_decode(result)
    return result.strip()


def infer_one(
    audio_path,
    encoder_session,
    decoder_session,
    vocab,
    task_code,
    input_size,
    n_mels,
    mel_filters_path,
    decoder_input_len,
    max_steps,
    show_tokens,
):
    x_mel = load_audio_to_mel(audio_path, input_size, n_mels, mel_filters_path)
    encoder_out = encoder_session.run(None, {"x": x_mel})[0]
    return run_decoder(
        decoder_session,
        encoder_out,
        vocab,
        task_code,
        decoder_input_len,
        max_steps,
        show_tokens=show_tokens,
    )


def collect_audio_files(source):
    source_path = Path(source)
    if source_path.is_file():
        return [source_path]

    if not source_path.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")

    exts = {".wav"}
    files = [p for p in source_path.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No wav files found in directory: {source}")
    return sorted(files)


def main():
    parser = argparse.ArgumentParser("Whisper ONNX Runtime inference")
    parser.add_argument("--encoder_model", type=str, default=DEFAULT_ENCODER, help="Whisper encoder ONNX model path")
    parser.add_argument("--decoder_model", type=str, default=DEFAULT_DECODER, help="Whisper decoder ONNX model path")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Wav path or wav directory")
    parser.add_argument("--task", type=str, default="en", choices=["en", "zh"], help="Recognition task")
    parser.add_argument("--vocab", type=str, default="", help="Optional vocab txt path, auto-selected by --task if empty")
    parser.add_argument("--mel_filters", type=str, default=DEFAULT_MEL_FILTER, help="Mel filter txt path")
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
        default=2000,
        help="Fallback mel width for dynamic encoder input shape",
    )
    parser.add_argument(
        "--decoder_input_size",
        type=int,
        default=12,
        help="Fallback token length for dynamic decoder token input",
    )
    parser.add_argument("--max_steps", type=int, default=128, help="Maximum autoregressive decoding steps")
    parser.add_argument("--show_tokens", action="store_true", help="Print token-level decoding details")
    args = parser.parse_args()

    providers = parse_providers(args.providers)
    encoder_session = build_session(args.encoder_model, providers)
    decoder_session = build_session(args.decoder_model, providers)

    encoder_input_size = get_encoder_width(encoder_session, args.input_size)
    decoder_input_len = get_decoder_token_len(decoder_session, args.decoder_input_size)

    if args.vocab:
        vocab_path = args.vocab
    else:
        vocab_path = DEFAULT_VOCAB_EN if args.task == "en" else DEFAULT_VOCAB_ZH

    task_code = 50259 if args.task == "en" else 50260
    vocab = read_vocab(vocab_path)

    audio_files = collect_audio_files(args.source)
    os.makedirs(args.save_dir, exist_ok=True)
    output_path = Path(args.save_dir) / "transcripts.txt"

    print(f"=== Running model: {args.encoder_model} + {args.decoder_model} ===")
    print(f"Task: {args.task}")
    print(f"Input mel size: {N_MELS}x{encoder_input_size}")
    print(f"Providers: {encoder_session.get_providers()}")

    lines = []
    for audio_file in audio_files:
        for i in range(6):
            if i == 1:
                start = time.perf_counter()
            text = infer_one(
                str(audio_file),
                encoder_session,
                decoder_session,
                vocab,
                task_code,
                encoder_input_size,
                N_MELS,
                args.mel_filters,
                decoder_input_len,
                args.max_steps,
                args.show_tokens,
            )
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
        print(f"    Average latency (whisper): {elapsed_ms:.2f} ms for a {audio_sec:.2f}s audio")
        print(f"    RTF: {elapsed_ms / (audio_sec * 1000.0):.4f}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()

'''
python ort_whisper.py \
  --encoder_model ../model/whisper_encoder_base_20s.onnx \
  --decoder_model ../model/whisper_decoder_base_20s.onnx \
  --source ../model/test_en.wav \
  --task en \
  --providers CPUExecutionProvider \
  --save_dir ./result_ort \
  --input_size 2000

python ort_whisper.py \
  --encoder_model ../model/whisper_encoder_base_20s.onnx \
  --decoder_model ../model/whisper_decoder_base_20s.onnx \
  --source ../model/test_zh.wav \
  --task zh \
  --providers CPUExecutionProvider \
  --save_dir ./result_ort \
  --input_size 2000
'''

