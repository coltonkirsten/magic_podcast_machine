from pathlib import Path
import json
import os
import subprocess
from typing import List, Dict

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


def main() -> None:
    client = load_client()
    script_path = Path("script.json")
    script = read_script(script_path)

    build_dir = Path("build")
    segments_dir = build_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

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

    final_path = build_dir / "podcast.mp3"
    print(f"[CONCAT] Stitching {len(segment_paths)} segments into {final_path.name}")
    concat_mp3_with_ffmpeg(segment_paths, final_path)
    print(f"Saved podcast to {final_path}")


if __name__ == "__main__":
    main()