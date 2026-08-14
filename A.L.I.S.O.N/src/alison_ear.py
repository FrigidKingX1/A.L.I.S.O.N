"""alison_ear.py -- Local Speech-to-Text & Microphone Perception for A.L.I.S.O.N.

Runs as a background thread inside the Core (Python 3.10) process so it can
reuse the already-loaded ``torch``/``ctranslate2`` environment and the shared
``alison_voice.Voice`` instance (which lazy-loads faster-whisper on CPU).

Two activation modes:
  * Push-To-Talk (PTT): the Core receives an ``start_listen`` IPC command
    (e.g. GUI ``Alt+V`` hotkey); the Ear records one utterance and queues it.
  * Ambient Wake-Word: when enabled, short ambient buffers are continuously
    transcribed; a match of r"\b(hey\s+)?alison\b" triggers capture of the
    following command. Toggle via the ``set_wakeword`` IPC command.

Transcripts are pushed to ``user_input_queue`` (same in-process queue the GUI
could also feed), so spoken commands receive identical self-model context and
Kokoro TTS responses as typed input. All audio/STT work is wrapped so a
missing microphone or model download failure degrades gracefully and never
interrupts the primary cognitive loop.
"""

import os
import re
import time

import numpy as np

from alison_ipc import TOPIC_EAR_STATE

WAKEWORD_RE = re.compile(r"\b(hey\s+)?alison\b", re.IGNORECASE)
SAMPLE_RATE = 16000


def _publish_state(ipc, state):
    try:
        if ipc is not None:
            ipc.publish_event(TOPIC_EAR_STATE, {"state": state})
    except Exception:
        pass


def _record_until_silence(stream, samplerate, silence_ms=600, max_ms=12000,
                          energy_thresh=0.012, start_pad_ms=150):
    """Pull audio from an open ``sd.InputStream`` until sustained silence.

    Returns a 1-D float32 numpy array at ``samplerate`` Hz, or an empty array
    if no speech was detected.
    """
    block = 1024
    buf = []
    silence_limit = int(silence_ms / 1000.0 * samplerate)
    start_pad = int(start_pad_ms / 1000.0 * samplerate)
    silence_frames = 0
    speech_detected = False
    t0 = time.time()
    try:
        while True:
            data, _ = stream.read(block)
            data = np.asarray(data, dtype=np.float32)[:, 0]
            buf.append(data)
            rms = float(np.sqrt(np.mean(data ** 2) + 1e-12))
            if rms > energy_thresh:
                speech_detected = True
                silence_frames = 0
            elif speech_detected:
                silence_frames += block
                if silence_frames >= silence_limit:
                    break
            if speech_detected and (len(buf) * block) > start_pad and (time.time() - t0) * 1000 > max_ms:
                break
            if (time.time() - t0) * 1000 > max_ms:
                break
    except Exception:
        pass
    if not buf:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(buf)
    return audio if speech_detected else np.zeros(0, dtype=np.float32)


def _transcribe(model, audio):
    if audio is None or len(audio) < SAMPLE_RATE * 0.2:
        return ""
    segments, _ = model.transcribe(audio, beam_size=5, language="en")
    return " ".join(seg.text for seg in segments).strip()


def run_ear(user_input_queue, ear_request_queue, wakeword_getter, ipc):
    """Background loop: PTT + (optional) ambient wake-word capture.

    Signature matches the threaded launch in ``alison_core._ipc_init``.
    """
    try:
        import sounddevice as sd
    except Exception as exc:
        _publish_state(ipc, "error")
        print(f"[EAR][warn] sounddevice unavailable ({exc}); mic disabled.")
        return

    try:
        from alison_voice import Voice
        voice = Voice()
        model = voice._ensure_stt()
    except Exception as exc:
        _publish_state(ipc, "error")
        print(f"[EAR][warn] Whisper STT init failed ({exc}); mic disabled.")
        return

    mic_device = None
    try:
        import alison_core
        _raw = alison_core.load_audio_config().get("input_device")
        mic_device = None if (_raw is None or _raw == -1) else _raw
    except Exception:
        pass

    print("[EAR] Listening (CPU Whisper, base.en, int8).")
    _publish_state(ipc, "idle")

    while True:
        try:
            # --- Push-To-Talk: GUI/hotkey requested one capture ---
            if not ear_request_queue.empty():
                try:
                    ear_request_queue.get_nowait()
                except Exception:
                    pass
                _publish_state(ipc, "listening")
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                    dtype="float32", blocksize=1024, device=mic_device) as stream:
                    audio = _record_until_silence(stream, SAMPLE_RATE)
                text = _transcribe(model, audio)
                if text:
                    user_input_queue.put(text)
                    print(f"\n  >>> [EAR][PTT] {text}")
                _publish_state(ipc, "idle")
                time.sleep(0.2)
                continue

            # --- Ambient wake-word sweep ---
            if wakeword_getter():
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                    dtype="float32", blocksize=1024, device=mic_device) as stream:
                    # Fixed short buffer for the wake-word scan.
                    probe, _ = stream.read(int(SAMPLE_RATE * 1.5))
                probe = np.asarray(probe, dtype=np.float32)[:, 0]
                probe_text = _transcribe(model, probe)
                if WAKEWORD_RE.search(probe_text or ""):
                    print(f"\n  >>> [EAR][WAKE] detected: {probe_text}")
                    _publish_state(ipc, "listening")
                    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                        dtype="float32", blocksize=1024, device=mic_device) as stream:
                        audio = _record_until_silence(stream, SAMPLE_RATE,
                                                      silence_ms=900)
                    text = _transcribe(model, audio)
                    if text:
                        user_input_queue.put(text)
                        print(f"\n  >>> [EAR][CMD] {text}")
                    _publish_state(ipc, "idle")
                time.sleep(0.1)
            else:
                time.sleep(0.5)
        except Exception as exc:
            print(f"[EAR][warn] loop error: {exc}")
            _publish_state(ipc, "idle")
            time.sleep(1.0)
