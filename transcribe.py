"""
Batch-transcribe Japanese audio/video to SRT subtitles using faster-whisper.

Usage:
  python transcribe.py <dir_or_file> [model] [--force]

Arguments:
  <dir_or_file>   Path to a media file or a directory (searched recursively).
                  Accepts .wav directly, or video files (.mp4, .mkv, .wmv,
                  .mov, .avi, .m4v) which are auto-converted to .wav first.
  [model]         Whisper model size (default: medium)
                  e.g. tiny, base, small, medium, large-v3
  --force         Re-generate .srt files even if they already exist

Notes:
  - Video files are converted to mono 16 kHz PCM WAV via ffmpeg. The
    resulting .wav is written next to the source and reused on later
    runs (conversion is skipped if the .wav already exists).
  - Model is loaded once and reused across files
  - Skips files that already have a .srt, unless --force is given
  - Progress is printed per segment with elapsed time and ETA
"""

from faster_whisper import WhisperModel
import sys
import pathlib
import subprocess
import time
import wave
from typing import Iterable

# ----------------------------
# Helpers
# ----------------------------

def get_wav_duration(path: pathlib.Path) -> float:
    """Return duration of a WAV file in seconds."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".wmv", ".mov", ".avi", ".m4v"}
MEDIA_EXTENSIONS = {".wav"} | VIDEO_EXTENSIONS

def iter_media(input_path: pathlib.Path) -> Iterable[pathlib.Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in MEDIA_EXTENSIONS:
            yield input_path
        return
    # directory: recursive search
    yield from sorted(
        p for p in input_path.rglob("*") if p.suffix.lower() in MEDIA_EXTENSIONS
    )

def convert_to_wav(source: pathlib.Path, dest: pathlib.Path) -> None:
    """Extract mono 16 kHz PCM WAV audio from a video (or other media) file via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )

# ----------------------------
# Args
# ----------------------------

if len(sys.argv) < 2:
    print("Usage: python transcribe.py <dir_or_file> [model] [--force]")
    sys.exit(1)

input_path = pathlib.Path(sys.argv[1]).expanduser().resolve()
model_size = "medium"
force = False

for arg in sys.argv[2:]:
    if arg == "--force":
        force = True
    else:
        model_size = arg  # allow model as second positional

# ----------------------------
# Setup model once
# ----------------------------

print(f"Input: {input_path}")
print(f"Model: {model_size}")
print("Loading model (once)...")

model = WhisperModel(
    model_size,
    device="cpu",
    compute_type="int8",
)

# ----------------------------
# Batch processing
# ----------------------------

media_files = list(iter_media(input_path))
if not media_files:
    print("No audio/video files found.")
    sys.exit(1)

# Resolve each media file to its target .wav, deduping so a video and an
# already-existing .wav for the same content aren't both processed.
seen_wavs = set()
targets = []
for m in media_files:
    wav_path = m if m.suffix.lower() == ".wav" else m.with_suffix(".wav")
    if wav_path in seen_wavs:
        continue
    seen_wavs.add(wav_path)
    targets.append((m, wav_path))

print(f"Found {len(targets)} file(s)\n")

batch_start = time.time()
ok = 0
skipped = 0
failed = 0

for i, (source_path, wav_path) in enumerate(targets, 1):
    output_path = wav_path.with_suffix(".srt")

    if output_path.exists() and not force:
        print(f"[{i}/{len(targets)}] SKIP (exists): {wav_path.name}")
        skipped += 1
        continue

    try:
        if not wav_path.exists():
            print(f"[{i}/{len(targets)}] Converting to WAV: {source_path.name} -> {wav_path.name}")
            convert_to_wav(source_path, wav_path)

        total_duration = get_wav_duration(wav_path)

        print(f"\n[{i}/{len(targets)}] Transcribing: {wav_path.name}")
        print(f"Duration: {mmss(total_duration)}")
        file_start = time.time()

        segments, info = model.transcribe(
            str(wav_path),
            language="ja",
            beam_size=5,
        )

        count = 0
        last_end = 0.0

        with output_path.open("w", encoding="utf-8") as f:
            for count, seg in enumerate(segments, 1):
                f.write(
                    f"{count}\n"
                    f"{srt_timestamp(seg.start)} --> {srt_timestamp(seg.end)}\n"
                    f"{seg.text.strip()}\n\n"
                )

                last_end = seg.end

                elapsed = time.time() - file_start
                progress = min(seg.end / total_duration, 1.0)
                speed = seg.end / elapsed if elapsed > 0 else 0
                eta = (total_duration - seg.end) / speed if speed > 0 else 0

                print(
                    f"\rSegments: {count:4d} | "
                    f"{mmss(seg.end)} / {mmss(total_duration)} "
                    f"({progress * 100:5.1f}%) | "
                    f"Elapsed: {elapsed:6.1f}s | "
                    f"ETA: {eta:6.1f}s",
                    end="",
                    flush=True
                )

        file_elapsed = time.time() - file_start
        ok += 1

        print("\nDone.")
        print(f"Wrote: {output_path}")
        print(f"Detected language: {info.language} (p={info.language_probability:.2f})")
        print(f"File time: {file_elapsed:.1f}s | Audio covered: {mmss(last_end)}")

    except Exception as e:
        failed += 1
        print(f"\nERROR on {wav_path.name}: {e}")

total_elapsed = time.time() - batch_start

print("\n=== Batch summary ===")
print(f"Processed: {len(targets)}")
print(f"Success  : {ok}")
print(f"Skipped  : {skipped}")
print(f"Failed   : {failed}")
print(f"Total time: {hms(total_elapsed)}")
