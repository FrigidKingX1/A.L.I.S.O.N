"""alison_voice.py -- Voice I/O for A.L.I.S.O.N.

Microphone capture + VAD + STT (faster-whisper) and TTS (Kokoro) with
sounddevice playback. Every heavy dependency is imported *lazily* so this
module imports cleanly even when faster-whisper / kokoro / sounddevice are
absent (e.g. a Core build that omits the voice extras). Always check
``Voice.available`` before calling the I/O methods.

The live microphone RMS is exposed via ``get_level()`` so the GUI can drive
the real ``u_audioRMS`` shader pulse (replacing the synthetic one).
"""

import os
import sys
import threading
import time

MODELS_DIR = None  # resolved lazily from alison_core if present


def _norm_device(dev):
    """Normalise a device selection: -1 or None both mean 'system default'."""
    if dev is None or dev == -1:
        return None
    return dev


def _resolve_models_dir():
    global MODELS_DIR
    if MODELS_DIR is not None:
        return MODELS_DIR
    try:
        import alison_core
        MODELS_DIR = alison_core.app_models_dir()
    except Exception:
        MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    return MODELS_DIR


class Voice:
    def __init__(self, stt_model="base", tts_voice="af_heart",
                 input_device=None, output_device=None):
        self.stt_model = stt_model
        self.tts_voice = tts_voice
        # Resolve device selection from persisted config when not explicitly set.
        if input_device is None or output_device is None:
            try:
                import alison_core
                cfg = alison_core.load_audio_config()
                if input_device is None:
                    input_device = cfg.get("input_device")
                if output_device is None:
                    output_device = cfg.get("output_device")
            except Exception:
                pass
        self.input_device = _norm_device(input_device)
        self.output_device = _norm_device(output_device)
        self._stt = None
        self._tts = None
        self._level = 0.0
        self._level_lock = threading.Lock()
        self._sample_rate = 16000
        self._meter_thread = None
        self._meter_running = False

    # ------------------------------------------------------------------
    @property
    def available(self):
        try:
            import sounddevice  # noqa: F401
            from faster_whisper import WhisperModel  # noqa: F401
            from kokoro import KPipeline  # noqa: F401
            return True
        except Exception:
            return False

    def _ensure_stt(self):
        if self._stt is None:
            from faster_whisper import WhisperModel
            # MUST stay on CPU: the Llama-3-8B Neocortex already consumes the
            # VRAM budget, so any CUDA allocation here would OOM the Core.
            self._stt = WhisperModel(
                self.stt_model,
                device="cpu",
                compute_type="int8",
                download_root=os.path.join(_resolve_models_dir(), "whisper"),
            )
        return self._stt

    def _ensure_tts(self):
        if self._tts is None:
            from kokoro import KPipeline
            self._tts = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        return self._tts

    # ------------------------------------------------------------------
    def start_level_meter(self):
        """Begin a background thread that samples mic RMS into ``get_level``."""
        if self._meter_running:
            return
        import sounddevice as sd

        def _run():
            self._meter_running = True
            try:
                with sd.InputStream(
                    channels=1, samplerate=self._sample_rate,
                    blocksize=1024, dtype="float32", device=self.input_device,
                ) as stream:
                    while self._meter_running:
                        data, _ = stream.read(1024)
                        import numpy as np
                        rms = float(np.sqrt(np.mean(np.square(data))))
                        with self._level_lock:
                            # Smooth toward the new sample.
                            self._level = 0.7 * self._level + 0.3 * min(1.0, rms * 8.0)
                        time.sleep(0.02)
            except Exception:
                self._meter_running = False

        self._meter_thread = threading.Thread(target=_run, daemon=True)
        self._meter_thread.start()

    def stop_level_meter(self):
        self._meter_running = False

    def get_level(self):
        with self._level_lock:
            return self._level

    # ------------------------------------------------------------------
    def listen_once(self, duration=5.0):
        """Record ``duration`` seconds from the mic and return transcript text."""
        import numpy as np
        import sounddevice as sd
        from faster_whisper import WhisperModel

        audio = sd.rec(
            int(duration * self._sample_rate),
            samplerate=self._sample_rate, channels=1, dtype="float32",
            device=self.input_device,
        )
        sd.wait()
        audio = np.squeeze(audio)
        model = self._ensure_stt()
        segments, _ = model.transcribe(audio, beam_size=5)
        return " ".join(seg.text for seg in segments).strip()

    def speak(self, text):
        """Synthesize ``text`` with Kokoro and play it via sounddevice."""
        import numpy as np
        import sounddevice as sd
        from kokoro import KPipeline

        pipeline = self._ensure_tts()
        for _, _, audio in pipeline(text, voice=self.tts_voice):
            if audio is not None and len(audio):
                sd.play(audio, samplerate=24000, device=self.output_device)
                sd.wait()
