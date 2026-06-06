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

### 1. Transcribe

```
python transcribe.py <dir_or_file> [model] [--force]
```

- `<dir_or_file>` — a media file or a directory (searched recursively). Accepts
  `.wav` directly, or video files (`.mp4`, `.mkv`, `.wmv`, `.mov`, `.avi`, `.m4v`)
- `[model]` — whisper model size: `tiny`, `base`, `small`, `medium` (default), `large-v3`
- `--force` — re-transcribe even if a `.srt` already exists

Video files are automatically converted to mono 16kHz PCM WAV via `ffmpeg` (required
on your `PATH`). The resulting `.wav` is written next to the source and reused on
later runs — conversion is skipped if a `.wav` already exists, so it's cheap to
re-run the script as new files show up.

Writes a `.srt` next to each file. Already-transcribed files (with an existing
`.srt`) are skipped unless `--force` is given, so it's safe to re-run on a
directory as new files are added.

### 2. Translate

```
python translate.py <dir_or_file> [--force]
```

- `<dir_or_file>` — a `.srt` file, or a directory (searched recursively)
- `--force` — re-translate even if a `.en.srt` already exists

Writes a `.en.srt` next to each `.srt`. Already-translated files are skipped, so it's safe to re-run on a directory as new subtitle files appear.

## Notes

- Requires `ffmpeg` on your `PATH` for video-to-WAV conversion.
- Both scripts load their model once and reuse it across a whole batch.
- Both print live progress (segments/lines processed, elapsed time, ETA) and a summary at the end (success/skipped/failed counts).
- Source language is hardcoded to Japanese (`language="ja"` for transcription, `Helsinki-NLP/opus-mt-ja-en` for translation).
