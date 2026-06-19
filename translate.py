"""
Batch-translate Japanese .srt subtitle files to English using NLLB-200.

Usage:
  python translate.py <dir_or_file> [model] [--force]

Arguments:
  <dir_or_file>   Path to a .srt file or a directory (searched recursively)
  [model]         Translation model: nllb (default) or marian
  --force         Re-generate .en.srt files even if they already exist

Notes:
  - Skips files that already end in .en.srt (and skips re-translating
    a file whose .en.srt output already exists, unless --force is given)
  - Model is loaded once and reused across files
  - Subtitle lines are translated in batches for speed, with live
    per-file progress, elapsed time, and ETA
"""

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
)
import sys
import pathlib
import time
import torch
from typing import Iterable

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
MARIAN_MODEL = "Helsinki-NLP/opus-mt-ja-en"
BATCH_SIZE = 8

# ----------------------------
# Helpers
# ----------------------------

def is_index(line: str) -> bool:
    return line.strip().isdigit()

def is_timestamp(line: str) -> bool:
    return "-->" in line

def hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def iter_srts(input_path: pathlib.Path) -> Iterable[pathlib.Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".srt" and not input_path.name.endswith(".en.srt"):
            yield input_path
        return
    # directory: recursive search, skipping already-translated output files
    yield from sorted(
        p for p in input_path.rglob("*.srt") if not p.name.endswith(".en.srt")
    )

def translate_batch_nllb(tokenizer, model, lines):
    inputs = tokenizer(lines, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_new_tokens=30,
            num_beams=5,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def translate_batch_marian(tokenizer, model, lines):
    inputs = tokenizer(lines, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, num_beams=5)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def translate_srt(tokenizer, model, translate_batch_fn, input_path: pathlib.Path, output_path: pathlib.Path) -> int:
    """Translate one .srt file to output_path. Returns the number of subtitle lines translated."""
    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)

    text_lines = [
        line for line in lines
        if line.strip() and not is_index(line) and not is_timestamp(line)
    ]
    total_lines = len(text_lines)
    print(f"Found {total_lines} subtitle line(s) to translate")

    translated_lines = []
    text_buffer = []
    translated_count = 0
    start_time = time.time()

    def flush_buffer():
        nonlocal translated_count
        translated = translate_batch_fn(tokenizer, model, text_buffer)
        for t in translated:
            translated_lines.append(t + "\n")
            translated_count += 1

            elapsed = time.time() - start_time
            progress = translated_count / total_lines if total_lines else 1.0
            speed = translated_count / elapsed if elapsed > 0 else 0
            eta = (total_lines - translated_count) / speed if speed > 0 else 0

            print(
                f"\rTranslated: {translated_count:4d}/{total_lines} "
                f"({progress * 100:5.1f}%) | "
                f"Elapsed: {elapsed:6.1f}s | "
                f"ETA: {eta:6.1f}s",
                end="",
                flush=True,
            )
        text_buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped or is_index(stripped) or is_timestamp(stripped):
            if text_buffer:
                flush_buffer()
            translated_lines.append(line)
        else:
            text_buffer.append(stripped)
            if len(text_buffer) >= BATCH_SIZE:
                flush_buffer()

    if text_buffer:
        flush_buffer()

    output_path.write_text("".join(translated_lines), encoding="utf-8")
    print()
    return translated_count

# ----------------------------
# Args
# ----------------------------

if len(sys.argv) < 2:
    print("Usage: python translate.py <dir_or_file> [model] [--force]")
    sys.exit(1)

input_path = pathlib.Path(sys.argv[1]).expanduser().resolve()
force = False
model_choice = "nllb"

for arg in sys.argv[2:]:
    if arg == "--force":
        force = True
    elif arg in ("nllb", "marian"):
        model_choice = arg
    else:
        print(f"Unknown argument: {arg}")
        sys.exit(1)

# ----------------------------
# Setup model once
# ----------------------------

print(f"Input: {input_path}")

if model_choice == "nllb":
    print(f"Loading NLLB-200 ({NLLB_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="jpn_Jpan")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    translate_batch_fn = translate_batch_nllb
else:
    print(f"Loading MarianMT ({MARIAN_MODEL})...")
    tokenizer = MarianTokenizer.from_pretrained(MARIAN_MODEL)
    model = MarianMTModel.from_pretrained(MARIAN_MODEL)
    translate_batch_fn = translate_batch_marian

model.eval()
model.to("cpu")

# ----------------------------
# Batch processing
# ----------------------------

srts = list(iter_srts(input_path))
if not srts:
    print("No .srt files found.")
    sys.exit(1)

print(f"Found {len(srts)} .srt file(s)\n")

batch_start = time.time()
ok = 0
skipped = 0
failed = 0

for i, srt_path in enumerate(srts, 1):
    output_path = srt_path.with_suffix(".en.srt")

    if output_path.exists() and not force:
        print(f"[{i}/{len(srts)}] SKIP (exists): {srt_path.name}")
        skipped += 1
        continue

    try:
        print(f"\n[{i}/{len(srts)}] Translating: {srt_path.name}")
        file_start = time.time()

        count = translate_srt(tokenizer, model, translate_batch_fn, srt_path, output_path)

        file_elapsed = time.time() - file_start
        ok += 1

        print(f"Wrote: {output_path}")
        print(f"File time: {file_elapsed:.1f}s | Lines translated: {count}")

    except Exception as e:
        failed += 1
        print(f"\nERROR on {srt_path.name}: {e}")

total_elapsed = time.time() - batch_start

print("\n=== Batch summary ===")
print(f"Processed: {len(srts)}")
print(f"Success  : {ok}")
print(f"Skipped  : {skipped}")
print(f"Failed   : {failed}")
print(f"Total time: {hms(total_elapsed)}")
