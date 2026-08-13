"""Validate LimbicToVocabBridge: prove different PC states produce distinct
affect vectors, and the trained bridge differentially biases Llama-3 tokens."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from alison_core import (
    AgentLM, CharTokenizer, LimbicSystem, LimbicToVocabBridge,
    Neocortex, calibrate_limbic_bridge, device
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

print("\n" + "=" * 70)
print("TEST 1: Distinct PC states produce distinct affect vectors (compute_affect)")
print("=" * 70)
states_list = [
    ("ANXIOUS (error+neg)", torch.tensor([0.0, 0.5, 0.0, 0.0, 0.8, 1.0], device=device)),
    ("CURIOUS (idle+pos)", torch.tensor([0.0, 0.2, 0.0, 0.8, 0.0, 0.0], device=device)),
    ("FATIGUED (idle+CPU)", torch.tensor([5.0, 0.9, 0.0, 0.0, 0.0, 0.0], device=device)),
    ("HAPPY (low+pos)", torch.tensor([0.0, 0.1, 0.0, 0.9, 0.1, 0.0], device=device)),
    ("PAIN (error+CPU)", torch.tensor([2.0, 0.8, 0.0, 0.0, 0.5, 0.5], device=device)),
]
affects = {}
for label, state in states_list:
    aff = ls.compute_affect(state)
    affects[label] = aff
    print(f"  {label:20s} -> norm={aff.norm().item():.4f}")

# Show cosine similarity matrix
print(f"\n  Cosine similarity matrix:")
keys = [k for k, _ in states_list]
for i, k1 in enumerate(keys):
    row = []
    for k2 in keys:
        sim = torch.cosine_similarity(affects[k1].view(-1), affects[k2].view(-1), dim=0).item()
        row.append(f"{sim:.3f}")
    print(f"  {k1[:12]:>12s}: {'  '.join(row)}")

print("\n" + "=" * 70)
print("TEST 2: Trained bridge produces different bias for different affects")
print("=" * 70)

# Tokenize Llama-3 words via llama-cpp tokenizer
word_pairs = [
    (" wait", " why"),
    (" danger", " explore"),
    (" stop", " how"),
    (" careful", " interesting"),
    (" urgent", " fascinating"),
    (" worried", " learn"),
    (" threat", " create"),
    (" attack", " curious"),
]
word_to_token = {}
for aw, cw in word_pairs:
    for w in [aw, cw]:
        if w not in word_to_token:
            raw = nc.model.tokenize(w.encode("utf-8"))
            word_to_token[w] = raw[0] if len(raw) > 0 else -1

# Test anxious vs curious affect
anxious_affect = affects["ANXIOUS (error+neg)"]
curious_affect = affects["CURIOUS (idle+pos)"]
fatigued_affect = affects["FATIGUED (idle+CPU)"]
happy_affect = affects["HAPPY (low+pos)"]

with torch.no_grad():
    af_a = anxious_affect.squeeze(0).squeeze(0).unsqueeze(0)
    af_c = curious_affect.squeeze(0).squeeze(0).unsqueeze(0)
    af_f = fatigued_affect.squeeze(0).squeeze(0).unsqueeze(0)
    af_h = happy_affect.squeeze(0).squeeze(0).unsqueeze(0)
    logits_a = bridge.proj(af_a)[0]
    logits_c = bridge.proj(af_c)[0]
    logits_f = bridge.proj(af_f)[0]
    logits_h = bridge.proj(af_h)[0]

print(f"  {'Token':>12s} | {'Anxious':>8s} | {'Curious':>8s} | {'Fatigued':>8s} | {'Happy':>8s}")
print(f"  {'-'*12} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
for aw, cw in word_pairs:
    t_aw = word_to_token.get(aw, -1)
    t_cw = word_to_token.get(cw, -1)
    if t_aw >= 0 and t_cw >= 0:
        l_a_w = logits_a[t_aw].item()
        l_a_c = logits_a[t_cw].item()
        l_c_w = logits_c[t_aw].item()
        l_c_c = logits_c[t_cw].item()
        l_f_w = logits_f[t_aw].item()
        l_f_c = logits_f[t_cw].item()
        l_h_w = logits_h[t_aw].item()
        l_h_c = logits_h[t_cw].item()
        warn = " <<<" if l_a_w > l_a_c else ""
        print(f"  {aw.strip():>12s} | {l_a_w:>+8.2f} | {l_c_w:>+8.2f} | {l_f_w:>+8.2f} | {l_h_w:>+8.2f}{warn}")
        warn = " <<<" if l_c_c > l_c_w else ""
        print(f"  {cw.strip():>12s} | {l_a_c:>+8.2f} | {l_c_c:>+8.2f} | {l_f_c:>+8.2f} | {l_h_c:>+8.2f}{warn}")

# Count direction correctness
correct_anxious = 0
correct_curious = 0
total = 0
for aw, cw in word_pairs:
    t_aw = word_to_token.get(aw, -1)
    t_cw = word_to_token.get(cw, -1)
    if t_aw >= 0 and t_cw >= 0:
        if logits_a[t_aw] > logits_a[t_cw]:
            correct_anxious += 1
        if logits_c[t_cw] > logits_c[t_aw]:
            correct_curious += 1
        total += 1

print(f"\nResults: Anxious boosts threat words {correct_anxious}/{total} ({100*correct_anxious/max(total,1):.0f}%)")
print(f"         Curious boosts explore words {correct_curious}/{total} ({100*correct_curious/max(total,1):.0f}%)")

print("\n" + "=" * 70)
print("TEST 3: All bridge logits are NOT identical across affects")
print("=" * 70)
diff_ac = (logits_a - logits_c).abs().mean().item()
diff_af = (logits_a - logits_f).abs().mean().item()
diff_ch = (logits_c - logits_h).abs().mean().item()
print(f"  Mean |logit_diff| anxious-vs-curious : {diff_ac:.4f}")
print(f"  Mean |logit_diff| anxious-vs-fatigued: {diff_af:.4f}")
print(f"  Mean |logit_diff| curious-vs-happy   : {diff_ch:.4f}")
print(f"\n  Without differentiation, all would be ~0.0000 (identical).")
print(f"  Differentiation confirmed: {'YES' if diff_ac > 0.001 else 'NO'}")

print("\n=== BRIDGE VALIDATION COMPLETE ===")
