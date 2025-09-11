from pathlib import Path
import json
import os
import subprocess
from typing import List, Dict
import wave
from array import array

import openai
from dotenv import load_dotenv


def load_client() -> openai.OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or your environment.")
    return openai.OpenAI(api_key=api_key)


def read_script(script_path: Path) -> List[Dict[str, str]]:
    with script_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("script.json must be a list of objects with 'host' and 'text'")
    return data


def host_to_voice(host: str) -> str:
    mapping = {
        "jamie": "marin",
        "alex": "cedar",
    }
    key = host.strip().lower()
    if key not in mapping:
        raise ValueError(f"No voice mapping for host: {host}")
    return mapping[key]


def synthesize_segment(client: openai.OpenAI, text: str, voice: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
    ) as response:
        response.stream_to_file(out_path)


def concat_mp3_with_ffmpeg(segment_paths: List[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_path.parent / "segments.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for p in segment_paths:
            # Use absolute paths to avoid relative path resolution issues
            f.write(f"file '{p.resolve().as_posix()}'\n")
    # Re-encode to ensure uniform codec params across segments
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(output_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))


def ffmpeg_convert_to_wav(src: Path, dst: Path, sample_rate: int = 44100, channels: int = 2) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))


def ffmpeg_generate_silence_wav(path: Path, duration_ms: int, sample_rate: int = 44100, channels: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl={'stereo' if channels == 2 else 'mono'}",
        "-t",
        str(duration_ms / 1000.0),
        "-c:a",
        "pcm_s16le",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))


def concat_wavs_python(input_paths: List[Path], output_path: Path) -> None:
    if not input_paths:
        raise ValueError("No WAV inputs to concatenate")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read params from first file
    with wave.open(str(input_paths[0]), "rb") as first:
        nchannels = first.getnchannels()
        sampwidth = first.getsampwidth()
        framerate = first.getframerate()
        comptype = first.getcomptype()
        compname = first.getcompname()

    with wave.open(str(output_path), "wb") as out_wav:
        out_wav.setnchannels(nchannels)
        out_wav.setsampwidth(sampwidth)
        out_wav.setframerate(framerate)
        out_wav.setcomptype(comptype, compname)

        for path in input_paths:
            with wave.open(str(path), "rb") as wav_in:
                if (
                    wav_in.getnchannels() != nchannels
                    or wav_in.getsampwidth() != sampwidth
                    or wav_in.getframerate() != framerate
                ):
                    raise ValueError(f"Incompatible WAV params for {path.name}")
                while True:
                    frames = wav_in.readframes(8192)
                    if not frames:
                        break
                    out_wav.writeframes(frames)


def _clip_int16(value: int) -> int:
    if value > 32767:
        return 32767
    if value < -32768:
        return -32768
    return value


def linear_crossfade_int16(
    tail_bytes: bytes,
    head_bytes: bytes,
    num_channels: int,
    fade_frames: int,
) -> bytes:
    if fade_frames <= 1:
        return tail_bytes
    # Convert to int16 arrays
    tail_arr = array("h")
    head_arr = array("h")
    tail_arr.frombytes(tail_bytes)
    head_arr.frombytes(head_bytes)
    total_samples = fade_frames * num_channels
    if len(tail_arr) < total_samples or len(head_arr) < total_samples:
        total_samples = min(len(tail_arr), len(head_arr))
        fade_frames = total_samples // num_channels
        total_samples = fade_frames * num_channels
    out_arr = array("h", [0] * total_samples)
    for sample_index in range(total_samples):
        frame_index = sample_index // num_channels
        t = frame_index / (fade_frames - 1)
        prev_gain = 1.0 - t
        next_gain = t
        mixed = tail_arr[sample_index] * prev_gain + head_arr[sample_index] * next_gain
        out_arr[sample_index] = _clip_int16(int(round(mixed)))
    return out_arr.tobytes()


def mix_concat_wavs_with_overlap(
    input_paths: List[Path],
    input_types: List[str],
    output_path: Path,
    speaking_gap_ms: int,
    overlap_ms: int,
) -> None:
    if not input_paths:
        raise ValueError("No WAV inputs to concatenate")
    if len(input_paths) != len(input_types):
        raise ValueError("input_paths and input_types length mismatch")

    # Read params from first file
    with wave.open(str(input_paths[0]), "rb") as first:
        nchannels = first.getnchannels()
        sampwidth = first.getsampwidth()
        framerate = first.getframerate()
        comptype = first.getcomptype()
        compname = first.getcompname()

    if sampwidth != 2:
        raise ValueError("Expected 16-bit PCM (sampwidth=2)")

    frame_size = sampwidth * nchannels

    def read_pcm(path: Path) -> bytes:
        with wave.open(str(path), "rb") as w:
            if (
                w.getnchannels() != nchannels
                or w.getsampwidth() != sampwidth
                or w.getframerate() != framerate
            ):
                raise ValueError(f"Incompatible WAV params for {path.name}")
            return w.readframes(w.getnframes())

    def silence_bytes(ms: int) -> bytes:
        frames = int(framerate * ms / 1000)
        return b"\x00" * (frames * frame_size)

    acc = bytearray(read_pcm(input_paths[0]))
    prev_type = input_types[0]

    overlap_frames = int(framerate * overlap_ms / 1000)
    overlap_bytes = overlap_frames * frame_size

    for i in range(1, len(input_paths)):
        curr_type = input_types[i]
        curr_pcm = read_pcm(input_paths[i])

        if prev_type == "speaking" and curr_type == "speaking":
            acc.extend(silence_bytes(speaking_gap_ms))
            acc.extend(curr_pcm)
        else:
            if overlap_bytes <= 0 or len(acc) < overlap_bytes or len(curr_pcm) < overlap_bytes:
                acc.extend(curr_pcm)
            else:
                tail = bytes(acc[-overlap_bytes:])
                head = curr_pcm[:overlap_bytes]
                mixed = linear_crossfade_int16(tail, head, num_channels=nchannels, fade_frames=overlap_frames)
                del acc[-overlap_bytes:]
                acc.extend(mixed)
                acc.extend(curr_pcm[overlap_bytes:])
        prev_type = curr_type

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as out_wav:
        out_wav.setnchannels(nchannels)
        out_wav.setsampwidth(sampwidth)
        out_wav.setframerate(framerate)
        out_wav.setcomptype(comptype, compname)
        out_wav.writeframes(bytes(acc))


def main() -> None:
    client = load_client()
    script_path = Path("script.json")
    script = read_script(script_path)

    build_dir = Path("build")
    segments_dir = build_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    wav_segments_dir = build_dir / "segments_wav"
    wav_segments_dir.mkdir(parents=True, exist_ok=True)

    segment_paths: List[Path] = []
    total = len(script)
    digits = len(str(total))
    for idx, item in enumerate(script, start=1):
        host = str(item.get("host", ""))
        text = str(item.get("text", ""))
        if not host or not text:
            continue
        voice = host_to_voice(host)
        filename = f"{str(idx).zfill(digits)}_{host}.mp3"
        out_path = segments_dir / filename
        print(f"[TTS] ({idx}/{total}) {host} -> {voice}: {out_path.name}")
        synthesize_segment(client, text, voice, out_path)
        segment_paths.append(out_path)

    # Convert all segments to uniform WAV for reliable concatenation
    wav_segment_paths: List[Path] = []
    for mp3_path in segment_paths:
        wav_path = wav_segments_dir / (mp3_path.stem + ".wav")
        ffmpeg_convert_to_wav(mp3_path, wav_path, sample_rate=44100, channels=2)
        wav_segment_paths.append(wav_path)

    # Prepare silence WAV asset: only used between speaking->speaking
    speaking_gap_ms = 600
    silence_dir = build_dir / "silence"
    silence_dir.mkdir(parents=True, exist_ok=True)
    silence_speaking_wav = silence_dir / f"silence_{speaking_gap_ms}ms.wav"
    ffmpeg_generate_silence_wav(silence_speaking_wav, speaking_gap_ms, sample_rate=44100, channels=2)

    # Build interleaved playlist (WAVs): gap ONLY between speaking -> speaking
    interleaved: List[Path] = []
    prev_type = None
    for i, item in enumerate(script):
        curr_type = str(item.get("type", "speaking")).strip().lower()
        if i > 0 and prev_type == "speaking" and curr_type == "speaking":
            interleaved.append(silence_speaking_wav)
        interleaved.append(wav_segment_paths[i])
        prev_type = curr_type

    # Concatenate with slight overlap for non speaking->speaking transitions
    final_wav = build_dir / "podcast.wav"
    speaking_types = [str(x.get("type", "speaking")).strip().lower() for x in script]
    print(
        f"[MIX] Building with overlap (speaking gap {speaking_gap_ms}ms, overlap 90ms) into {final_wav.name}"
    )
    mix_concat_wavs_with_overlap(
        input_paths=wav_segment_paths,
        input_types=speaking_types,
        output_path=final_wav,
        speaking_gap_ms=speaking_gap_ms,
        overlap_ms=90,
    )

    final_mp3 = build_dir / "podcast.mp3"
    cmd_encode = [
        "ffmpeg",
        "-y",
        "-i",
        str(final_wav),
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(final_mp3),
    ]
    proc = subprocess.run(cmd_encode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    print(f"Saved podcast to {final_mp3}")


if __name__ == "__main__":
    main()