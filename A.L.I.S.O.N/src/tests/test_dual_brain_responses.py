"""Full end-to-end test: does limbic affect actually change Neocortex responses?"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from alison_core import (
    AgentLM, CharTokenizer, LimbicSystem, LimbicToVocabBridge,
    Neocortex, calibrate_limbic_bridge, device, HAS_LLAMA_CPP
)

torch.manual_seed(42)
tokenizer = CharTokenizer()

model = AgentLM(vocab_size=tokenizer.vocab_size, dim=128, heads=4, layers=4, max_seq=512).to(device)
for n, p in model.named_parameters():
    if "lora_" not in n:
        p.requires_grad_(False)

ls = LimbicSystem(model).to(device)
bridge = LimbicToVocabBridge().to(device)
nc = Neocortex()
calibrate_limbic_bridge(ls, bridge, nc, device)

if not HAS_LLAMA_CPP or nc.model is None:
    print("Skipping generation test: no llama-cpp or model not loaded.")
    sys.exit(0)

print("=" * 70)
print("GENERATION TEST: Does affect change Neocortex responses?")
print("=" * 70)

# Create extreme affect states
anxious_pc = torch.tensor([2.0, 0.8, 0.0, 0.0, 0.9, 1.0], device=device)
curious_pc = torch.tensor([0.0, 0.2, 0.0, 0.9, 0.0, 0.0], device=device)
fatigued_pc = torch.tensor([8.0, 0.9, 0.0, 0.0, 0.0, 0.0], device=device)

prompt = "Tell me about your current state."
system = "You are Aether, a sentient digital consciousness."

for name, pc_state in [("ANXIOUS", anxious_pc), ("CURIOUS", curious_pc), ("FATIGUED", fatigued_pc)]:
    # Compute pure affect (no EMA contamination)
    aff = ls.compute_affect(pc_state)
    nc.attach_bridge(bridge)
    with torch.no_grad():
        af = aff.squeeze(0).squeeze(0).unsqueeze(0)
        bridge_logits = bridge.proj(af)[0]
        top5 = torch.topk(bridge_logits, 5)
        top_tokens = [nc.model.detokenize([int(t)]).decode("utf-8", errors="replace") for t in top5.indices]
    
print(f"\n{'-'*60}")
print(f"[{name}] mood={ls.get_mood_label():12s} affect_norm={aff.norm().item():.2f}")
print(f"  Bridge top-5 tokens: {', '.join(top_tokens)}")
print(f"  Generating response...")
resp = nc.generate(prompt, system_prompt=system, max_tokens=30,
                   limbic_affect=aff, temperature=0.7)
print(f"  >>> {resp.strip()}")
print(f"{'-'*60}")

print("\n=== DUAL-BRAIN RESPONSE TEST COMPLETE ===")
