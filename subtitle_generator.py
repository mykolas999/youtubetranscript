#!/usr/bin/env python3
"""
Video to SRT subtitle generator with word-by-word appearing subtitles for reels.

Usage:
    python subtitle_generator.py video.mp4
    python subtitle_generator.py video.mp4 --output subtitles.srt
    python subtitle_generator.py video.mp4 --words-per-line 4 --language lt
    python subtitle_generator.py video.mp4 --backend local --model-size small
"""

import argparse
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from datetime import timedelta


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm"""
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def check_ffmpeg() -> None:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if result.returncode != 0:
        print("ERROR: ffmpeg not found. Install it with: sudo apt install ffmpeg", file=sys.stderr)
        sys.exit(1)


def extract_audio(video_path: str, audio_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")


def transcribe_openai_api(audio_path: str, language: str | None) -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    with open(audio_path, "rb") as f:
        kwargs: dict = {
            "model": "whisper-1",
            "file": f,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }
        if language:
            kwargs["language"] = language

        transcript = client.audio.transcriptions.create(**kwargs)

    words = []
    for w in transcript.words:
        words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    return words


def transcribe_local_whisper(audio_path: str, model_size: str, language: str | None) -> list[dict]:
    try:
        import whisper
    except ImportError:
        print("ERROR: whisper package not installed. Run: pip install openai-whisper", file=sys.stderr)
        sys.exit(1)

    print(f"Loading local Whisper model '{model_size}'...", flush=True)
    model = whisper.load_model(model_size)

    kwargs: dict = {"word_timestamps": True}
    if language:
        kwargs["language"] = language

    print("Transcribing...", flush=True)
    result = model.transcribe(audio_path, **kwargs)

    words = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"],
            })
    return words


def build_srt(words: list[dict], words_per_line: int) -> str:
    if not words:
        return ""

    srt_parts: list[str] = []
    entry_num = 1

    groups = [words[i:i + words_per_line] for i in range(0, len(words), words_per_line)]

    for group in groups:
        for i, word in enumerate(group):
            start = word["start"]

            if i < len(group) - 1:
                end = group[i + 1]["start"]
            else:
                end = word["end"]

            if end - start < 0.05:
                end = start + 0.05

            text = " ".join(w["word"] for w in group[: i + 1])

            srt_parts.append(
                f"{entry_num}\n"
                f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
                f"{text}\n"
            )
            entry_num += 1

    return "\n".join(srt_parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reels-style word-by-word SRT subtitles from a video file."
    )
    parser.add_argument("video", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output SRT file path")
    parser.add_argument("--words-per-line", type=int, default=5, metavar="N",
        help="Words per subtitle line group (default: 5)")
    parser.add_argument("--language", default=None, metavar="LANG",
        help="Audio language code, e.g. lt, en, ru (auto-detected if omitted)")
    parser.add_argument("--backend", choices=["api", "local"], default="api",
        help="'api' uses OpenAI Whisper API (needs OPENAI_API_KEY), 'local' uses local model")
    parser.add_argument("--model-size",
        choices=["tiny", "base", "small", "medium", "large", "turbo"], default="base",
        help="Local Whisper model size (only with --backend local, default: base)")
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(Path(video_path).with_suffix(".srt"))

    check_ffmpeg()

    print(f"Extracting audio from '{video_path}'...", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        extract_audio(video_path, audio_path)
        print("Audio extracted.", flush=True)

        if args.backend == "api":
            print("Transcribing via OpenAI Whisper API...", flush=True)
            words = transcribe_openai_api(audio_path, args.language)
        else:
            words = transcribe_local_whisper(audio_path, args.model_size, args.language)

        print(f"Transcription complete - {len(words)} words detected.", flush=True)

        srt_content = build_srt(words, args.words_per_line)

        Path(output_path).write_text(srt_content, encoding="utf-8")
        print(f"SRT file saved: {output_path}", flush=True)

    finally:
        if Path(audio_path).exists():
            Path(audio_path).unlink()


if __name__ == "__main__":
    main()
