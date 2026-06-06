"""
Compare JA->EN subtitle translations from MarianMT vs NLLB-200, side by side.

Usage:
  python compare_translations.py <input.srt> [--limit N]

Writes <input>.compare.txt with each subtitle block shown as:

  [12] 00:01:32,000 --> 00:01:35,500
  JA:       すみません、もう一度お願いします
  MarianMT: Excuse me, please again
  NLLB:     Sorry, could you say that one more time?

Notes:
  - Both models translate the exact same set of lines, so the
    outputs line up and are directly comparable block-by-block.
  - Use --limit N to compare only the first N subtitle blocks
    (useful for a quick spot-check on long files).
"""

import sys
import pathlib
import time
import torch
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

MARIAN_MODEL = "Helsinki-NLP/opus-mt-ja-en"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
BATCH_SIZE = 8

# ----------------------------
# SRT parsing
# ----------------------------

def parse_srt_blocks(path: pathlib.Path):
    """Return a list of (index, timestamp, text) for each subtitle block."""
    blocks = []
    current = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            current.append(line)
        else:
            if current:
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)

    parsed = []
    for block in blocks:
        if len(block) < 3:
            continue
        index, timestamp = block[0], block[1]
        text = " ".join(block[2:]).strip()
        parsed.append((index, timestamp, text))
    return parsed

# ----------------------------
# Models
# ----------------------------

def load_marian():
    print(f"Loading MarianMT ({MARIAN_MODEL})...")
    tokenizer = MarianTokenizer.from_pretrained(MARIAN_MODEL)
    model = MarianMTModel.from_pretrained(MARIAN_MODEL)
    model.eval()
    model.to("cpu")
    return tokenizer, model

def load_nllb():
    print(f"Loading NLLB-200 ({NLLB_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="jpn_Jpan")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    model.eval()
    model.to("cpu")
    return tokenizer, model

def translate_marian(tokenizer, model, texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, num_beams=5)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def translate_nllb(tokenizer, model, texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=512,
            num_beams=5,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def translate_all(label, tokenizer, model, translate_fn, texts):
    """Translate all texts in batches, printing live progress."""
    results = []
    total = len(texts)
    start = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        results.extend(translate_fn(tokenizer, model, batch))

        done = len(results)
        elapsed = time.time() - start
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0

        print(
            f"\r{label:8s}: {done:4d}/{total} "
            f"({done / total * 100:5.1f}%) | "
            f"Elapsed: {elapsed:6.1f}s | "
            f"ETA: {eta:6.1f}s",
            end="",
            flush=True,
        )

    print()
    return results

# ----------------------------
# Args
# ----------------------------

if len(sys.argv) < 2:
    print("Usage: python compare_translations.py <input.srt> [--limit N]")
    sys.exit(1)

input_srt = pathlib.Path(sys.argv[1]).expanduser().resolve()
output_path = input_srt.with_suffix(".compare.txt")

limit = None
args = sys.argv[2:]
if "--limit" in args:
    limit = int(args[args.index("--limit") + 1])

# ----------------------------
# Load subtitles
# ----------------------------

print(f"Input: {input_srt}")
blocks = parse_srt_blocks(input_srt)
if limit:
    blocks = blocks[:limit]

texts = [text for _, _, text in blocks]
print(f"Found {len(texts)} subtitle line(s) to translate\n")

if not texts:
    print("No subtitle lines found.")
    sys.exit(1)

# ----------------------------
# Translate with both models
# ----------------------------

marian_tokenizer, marian_model = load_marian()
marian_results = translate_all(
    "MarianMT", marian_tokenizer, marian_model, translate_marian, texts
)

nllb_tokenizer, nllb_model = load_nllb()
nllb_results = translate_all(
    "NLLB", nllb_tokenizer, nllb_model, translate_nllb, texts
)

# ----------------------------
# Write side-by-side comparison
# ----------------------------

with output_path.open("w", encoding="utf-8") as f:
    for (index, timestamp, ja_text), marian_text, nllb_text in zip(
        blocks, marian_results, nllb_results
    ):
        f.write(f"[{index}] {timestamp}\n")
        f.write(f"JA:       {ja_text}\n")
        f.write(f"MarianMT: {marian_text}\n")
        f.write(f"NLLB:     {nllb_text}\n")
        f.write("\n")

print(f"\nWrote comparison to: {output_path}")
