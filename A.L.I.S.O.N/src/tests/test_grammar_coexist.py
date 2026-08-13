"""Phase 34.1 verification: GBNF grammar + limbic logits bias coexistence."""
import sys, os, json, torch
sys.path.insert(0, r"E:\ClaudeATHome\Projects\Coding Projects")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from alison_core import Neocortex, LimbicToVocabBridge, JSON_GRAMMAR, device

nc = Neocortex()
bridge = LimbicToVocabBridge().to(device)
nc.attach_bridge(bridge)
assert nc.model is not None

affect = torch.zeros(1, 1, 128, device=device)
import time
start = time.time()
resp = nc.generate(
    "Recent chats: user said the project crashed. Current persona: "
    '{"traits": "curious, analytical", "mood": "neutral", "relationship": "acquaintance"}. '
    "Update traits, mood, relationship. Output ONLY a valid JSON object.",
    system_prompt="You are Aether's prefrontal cortex. Output ONLY a valid JSON object.",
    max_tokens=150,
    temperature=0.8,
    limbic_affect=affect,
    grammar=JSON_GRAMMAR,
)
elapsed = time.time() - start
print(f"GENERATION ({elapsed:.1f}s): {resp!r}")
parsed = json.loads(resp.strip())
print(f"JSON VALID: {parsed}")
assert isinstance(parsed, dict), "not a JSON object"
print("COEXISTENCE OK: grammar + limbic logits_processor applied together")