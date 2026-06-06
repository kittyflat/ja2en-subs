# ja2en-subs

Pipeline for turning Japanese video into English subtitles:

1. **Transcribe** Japanese audio to `.srt` using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
2. **Translate** the resulting `.srt` from Japanese to English using [MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en)

## Setup

```
python -m venv ~/.venvs/ja2en-subs-env
source ~/.venvs/ja2en-subs-env/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Convert video to WAV (mono, 16kHz PCM)

`faster-whisper` works best on mono 16kHz PCM WAV. Convert your source videos first:

```
for f in *.wmv; do
  ffmpeg -i "$f" -vn -ac 1 -ar 16000 -c:a pcm_s16le "${f%.wmv}.wav"
done
```

### 2. Transcribe

```
python transcribe.py <dir_or_file> [model] [--force]
```

- `<dir_or_file>` — a `.wav` file, or a directory (searched recursively)
- `[model]` — whisper model size: `tiny`, `base`, `small`, `medium` (default), `large-v3`
- `--force` — re-transcribe even if a `.srt` already exists

Writes a `.srt` next to each `.wav`. Already-transcribed files (with an existing `.srt`) are skipped, so it's safe to re-run on a directory as new files are added.

### 3. Translate

```
python translate.py <dir_or_file> [--force]
```

- `<dir_or_file>` — a `.srt` file, or a directory (searched recursively)
- `--force` — re-translate even if a `.en.srt` already exists

Writes a `.en.srt` next to each `.srt`. Already-translated files are skipped, so it's safe to re-run on a directory as new subtitle files appear.

## Notes

- Both scripts load their model once and reuse it across a whole batch.
- Both print live progress (segments/lines processed, elapsed time, ETA) and a summary at the end (success/skipped/failed counts).
- Source language is hardcoded to Japanese (`language="ja"` for transcription, `Helsinki-NLP/opus-mt-ja-en` for translation).
