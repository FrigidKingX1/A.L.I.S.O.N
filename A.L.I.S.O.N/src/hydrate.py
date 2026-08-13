"""A.L.I.S.O.N. -- Phase 3 lazy weight hydration.

First-run validator/downloader for the external model weights the engine
actually consumes. Honors the engine's loader (`alison_core.py` -> llama_cpp
 loading `models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`).

By default weights land in the script-local `models/` directory (which is
exactly where the engine looks). Override with the `ALISON_MODELS_DIR`
environment variable (e.g. `%LOCALAPPDATA%\\ALISON\\models`).

The DeBERTa-v3 entry is intentionally DISABLED: the current engine has no
DeBERTa consumer (it is a llama.cpp dual-brain). Enable it only if you later
wire a secondary transformer pipeline.
"""

import os
import sys
import argparse
import hashlib

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    sys.exit("[hydrate] huggingface_hub is required: pip install huggingface_hub")


# Resolve the destination directory.
def resolve_models_dir():
    env = os.environ.get("ALISON_MODELS_DIR")
    if env:
        return os.path.expandvars(env)
    if getattr(sys, "frozen", False):
        # Installed builds run from a read-only location (e.g. Program Files);
        # weights must live in a user-writable path that matches the engine.
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "A.L.I.S.O.N.", "models")
    # Script-local `models/` -- matches alison_core.py's dev loader.
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


MODELS_DIR = resolve_models_dir()

# ---------------------------------------------------------------------------
# Hydration manifest. Enable/extend as the engine's needs grow.
# ---------------------------------------------------------------------------
HYDRATION_MANIFEST = {
    # 1. Neocortex -- llama.cpp Q4_K_M GGUF (the engine's actual model).
    "neocortex": {
        "enabled": True,
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "expected_sha256": None,  # set to pin; None = accept hub-verified file
        "local_name": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    },
    # 2. DeBERTa-v3 -- DISABLED (no engine consumer yet).
    #    repo_id: cross-encoder/nli-deberta-v3-xsmall
    #    Enable only after integrating a DeBERTa sensory module.
    "sensory_deberta_v3": {
        "enabled": False,
        "repo_id": "cross-encoder/nli-deberta-v3-xsmall",
        "filename": "pytorch_model.bin",
        "expected_sha256": None,
        "local_name": "deberta_v3_xsmall/pytorch_model.bin",
    },
    # 3. Voice STT model -- DISABLED by default. Enable (set enabled: True) only
    #    after installing faster-whisper + sounddevice in the runtime. Kokoro
    #    TTS voices ship with the pip package, so no download is required.
    "voice_whisper": {
        "enabled": False,
        "repo_id": "Systran/faster-whisper-base",
        "filename": "model.bin",
        "expected_sha256": None,
        "local_name": "whisper/faster-whisper-base/model.bin",
    },
}


def _sha256(path, expected):
    if expected is None:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def hydrate_one(key, spec, force=False):
    if not spec.get("enabled", False):
        print(f"[hydrate] skip (disabled): {key}")
        return True
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = os.path.join(MODELS_DIR, spec["local_name"])
    if os.path.exists(dest) and not force:
        if _sha256(dest, spec.get("expected_sha256")):
            print(f"[hydrate] present & valid: {dest}")
            return True
        print(f"[hydrate] checksum mismatch, re-downloading: {dest}")
    print(f"[hydrate] fetching {key} from {spec['repo_id']}/{spec['filename']} ...")
    try:
        hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[hydrate] ERROR fetching {key}: {exc}")
        return False
    # hf_hub_download writes into local_dir/<filename>; ensure local_name path.
    if spec["local_name"] != spec["filename"]:
        src = os.path.join(MODELS_DIR, spec["filename"])
        if os.path.exists(src) and not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.replace(src, dest)
    if not _sha256(dest, spec.get("expected_sha256")):
        print(f"[hydrate] WARNING: {dest} failed SHA-256 validation")
        return False
    print(f"[hydrate] OK: {dest}")
    return True


def main():
    p = argparse.ArgumentParser(description="A.L.I.S.O.N. lazy weight hydration")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--check", action="store_true",
                   help="verify present files only; do not download")
    args = p.parse_args()

    if args.check and args.force:
        sys.exit("--check and --force are mutually exclusive")

    print(f"[hydrate] models dir = {MODELS_DIR}")
    ok = True
    for key, spec in HYDRATION_MANIFEST.items():
        if args.check:
            dest = os.path.join(MODELS_DIR, spec["local_name"])
            present = os.path.exists(dest) and _sha256(dest, spec.get("expected_sha256"))
            print(f"[hydrate] {'OK ' if present else 'MISSING'} {key}: {dest}")
            ok = ok and (present or not spec.get("enabled", False))
        else:
            ok = hydrate_one(key, spec, force=args.force) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
