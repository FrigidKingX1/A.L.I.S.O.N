"""Unit check: deterministic self-model routing (zero-LLM)."""
import sys, os, torch
sys.path.insert(0, r"E:\ClaudeATHome\Projects\Coding Projects")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import alison_core as ica

ctx_error = "Window: terminal\nActivity: system error message\nText: FATAL ERROR: segmentation fault in module core.py at line 42"
ctx_video = "Window: YouTube\nActivity: watching a video stream\nText: Exploring a server that doesn't exist - cool tech video"

print("== Deterministic self-model (error context) ==")
mood1 = ica.update_self_model_deterministic(ica.limbic_system, ica.world, ctx_error)
print("== Deterministic self-model (video context) ==")
mood2 = ica.update_self_model_deterministic(ica.limbic_system, ica.world, ctx_video)
print("== Deterministic self-model (fallback, no screen) ==")
mood3 = ica.update_self_model_deterministic(ica.limbic_system, ica.world, None)

print(f"\nMOODS: {mood1} -> {mood2} -> {mood3}")
v6 = ica.limbic_system._get_v6()
print("V6 after routing:", [round(x, 3) for x in v6.tolist()])
assert mood1 in ("HUNGRY", "PAIN", "FATIGUED", "CURIOUS", "ANXIOUS", "ALTRUISTIC")
assert isinstance(mood2, str) and isinstance(mood3, str)
print("DETERMINISTIC SELF-MODEL: PASS")