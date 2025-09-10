from pathlib import Path
import json
import os
import subprocess
from typing import List, Dict
import wave

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

    # Concatenate into final WAV then transcode to MP3
    final_wav = build_dir / "podcast.wav"
    print(f"[CONCAT] Stitching {len(interleaved)} WAV items into {final_wav.name}")
    concat_wavs_python(interleaved, final_wav)

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