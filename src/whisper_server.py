"""
Persistent faster-whisper transcription server.

Protocol (line-based JSON over stdin/stdout):
  Request:  {"wav": "/path/to/audio.wav"}
  Response: {"text": "transcribed text"} or {"text": "", "error": "..."}
"""

import json
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Force all logging to stderr to protect our JSON stdout protocol.
import logging
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
_original_stream_handler_init = logging.StreamHandler.__init__
def _patched_stream_handler_init(self, stream=None):
    _original_stream_handler_init(self, sys.stderr)
logging.StreamHandler.__init__ = _patched_stream_handler_init

from faster_whisper import WhisperModel


def load_hotwords(hotwords_file):
    """Read hotwords file and return as comma-separated prompt string."""
    try:
        if os.path.exists(hotwords_file):
            with open(hotwords_file, "r") as f:
                words = [line.strip() for line in f if line.strip()]
            if words:
                return ", ".join(words)
    except Exception as e:
        sys.stderr.write(f"Error reading hotwords: {e}\n")
        sys.stderr.flush()
    return None


def main():
    model_name = os.environ.get("WHISPER_MODEL", "base")
    device = os.environ.get("WHISPER_DEVICE", "auto")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "auto")
    hotwords_file = os.environ.get("HOTWORDS_FILE", "hotwords.txt")

    sys.stderr.write(f"Loading faster-whisper model '{model_name}' (device={device}, compute={compute_type})...\n")
    sys.stderr.flush()

    initial_prompt = load_hotwords(hotwords_file)
    if initial_prompt:
        sys.stderr.write(f"Loaded hotwords: {initial_prompt}\n")
        sys.stderr.flush()

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )

    sys.stderr.write("faster-whisper model loaded. Ready.\n")
    sys.stderr.flush()

    # Signal readiness to the Node process
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            wav_path = req["wav"]
            context = req.get("context", "")

            # Re-read hotwords on every request so the web UI's additions
            # take effect without restarting the whisper server.
            hotwords_prompt = load_hotwords(hotwords_file)

            # Build initial_prompt: hotwords + previous transcription context.
            prompt_parts = []
            if hotwords_prompt:
                prompt_parts.append(hotwords_prompt)
            if context:
                prompt_parts.append(context)
            initial_prompt = ". ".join(prompt_parts) if prompt_parts else None

            segments, info = model.transcribe(
                wav_path,
                language="en",
                initial_prompt=initial_prompt,
                vad_filter=False,
                beam_size=5,
            )

            # Combine all segments into one text and compute average confidence
            seg_list = list(segments)
            text = " ".join(seg.text.strip() for seg in seg_list)
            text = text.strip()

            # Average probability across segments as confidence score.
            # faster-whisper Segment uses .avg_logprob (no underscore before prob)
            confidence = 0.0
            if seg_list:
                import math
                try:
                    confidence = sum(seg.avg_logprob for seg in seg_list) / len(seg_list)
                    confidence = math.exp(confidence)
                    confidence = max(0.0, min(1.0, confidence))
                except AttributeError:
                    confidence = 0.0

            print(json.dumps({"text": text, "confidence": round(confidence, 3)}), flush=True)

        except Exception as e:
            sys.stderr.write(f"Transcription error: {e}\n")
            sys.stderr.flush()
            print(json.dumps({"text": "", "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
