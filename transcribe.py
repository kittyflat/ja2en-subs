"""
Batch-transcribe Japanese audio to SRT subtitles using faster-whisper.

Usage:
  python transcribe.py <dir_or_file> [model] [--force]

Arguments:
  <dir_or_file>   Path to a .wav file or a directory (searched recursively)
  [model]         Whisper model size (default: medium)
                  e.g. tiny, base, small, medium, large-v3
  --force         Re-generate .srt files even if they already exist

Notes:
  - WAV input should be mono, 16 kHz PCM for best results
  - Model is loaded once and reused across files
  - Skips files that already have a .srt, unless --force is given
  - Progress is printed per segment with elapsed time and ETA
"""

from faster_whisper import WhisperModel
import sys
import pathlib
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

def iter_wavs(input_path: pathlib.Path) -> Iterable[pathlib.Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".wav":
            yield input_path
        return
    # directory: recursive search
    yield from sorted(input_path.rglob("*.wav"))

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

wavs = list(iter_wavs(input_path))
if not wavs:
    print("No .wav files found.")
    sys.exit(1)

print(f"Found {len(wavs)} .wav file(s)\n")

batch_start = time.time()
ok = 0
skipped = 0
failed = 0

for i, wav_path in enumerate(wavs, 1):
    output_path = wav_path.with_suffix(".srt")

    if output_path.exists() and not force:
        print(f"[{i}/{len(wavs)}] SKIP (exists): {wav_path.name}")
        skipped += 1
        continue

    try:
        total_duration = get_wav_duration(wav_path)

        print(f"\n[{i}/{len(wavs)}] Transcribing: {wav_path.name}")
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
print(f"Processed: {len(wavs)}")
print(f"Success  : {ok}")
print(f"Skipped  : {skipped}")
print(f"Failed   : {failed}")
print(f"Total time: {hms(total_elapsed)}")
