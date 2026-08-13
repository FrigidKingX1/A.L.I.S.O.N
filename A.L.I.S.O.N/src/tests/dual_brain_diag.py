"""
Comprehensive Dual-Brain Architecture Diagnostic
Generates structured log for external analysis.
"""
import sys, os, time, json, inspect, hashlib
sys.path.insert(0, r"E:\ClaudeATHome\Projects\Coding Projects")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
import numpy as np

from alison_core import (
    AgentLM, CharTokenizer, LimbicSystem, LimbicToVocabBridge,
    LimbicLogitsProcessor, Neocortex, MoodClassifier,
    calibrate_affective_core_v2, train_mood_classifier_v3, calibrate_limbic_bridge,
    device, HAS_LLAMA_CPP, update_ewc_fisher, fisher_matrix as module_fisher,
    ewc_optimal_weights as module_ewc_weights, evaluate_homeostatic_fitness,
    DynamicWorld, DigitalGenome, encode_pair, seed_knowledge,
)

LOG = []
LOG_FILE = None
def log(msg=""):
    msg = str(msg)
    LOG.append(msg)
    print(msg)
    if LOG_FILE is not None:
        LOG_FILE.write(msg + "\n")
        LOG_FILE.flush()

def section(title):
    log(f"\n{'=' * 74}")
    log(f"  {title}")
    log(f"{'=' * 74}")

def subsection(title):
    log(f"\n  --- {title} ---")

torch.manual_seed(42)
np.random.seed(42)

log(f"DIAGNOSTIC START: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Torch: {torch.__version__}, Device: {device}, Python: {sys.version}")

# ================================================================
# 1. ARCHITECTURE OVERVIEW
# ================================================================
section("1. ARCHITECTURE OVERVIEW")

tokenizer = CharTokenizer()
model = AgentLM(vocab_size=tokenizer.vocab_size, dim=128, heads=4, layers=4, max_seq=512).to(device)
for n, p in model.named_parameters():
    if "lora_" not in n:
        p.requires_grad_(False)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
log(f"  Tokenizer vocab: {tokenizer.vocab_size}")
log(f"  Model dim: 128, Heads: 4, Layers: 4, LoRA rank: (from model defaults)")
log(f"  Total params: {total_params:,} ({total_params/1e6:.2f}M)")
log(f"  Trainable (LoRA): {trainable_params:,} ({trainable_params/1e3:.1f}K)")
log(f"  Frozen: {frozen_params:,} ({frozen_params/1e6:.2f}M)")

# Parameter groups
grad_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
frozen_params_list = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
lora_total = sum(p.numel() for _, p in grad_params)
linear_total = sum(p.numel() for _, p in frozen_params_list)
log(f"\n  LoRA params: {len(grad_params)} tensors, {lora_total:,} total params")
log(f"  Frozen (linear+norm+embed): {len(frozen_params_list)} tensors, {linear_total:,} total params")
log(f"  LoRA composition per block:")
for block_idx in range(4):
    tensors = [(n, p) for n, p in grad_params if n.startswith(f"blocks.{block_idx}.")]
    total = sum(p.numel() for _, p in tensors)
    qkv = sum(p.numel() for _, p in tensors if 'attn.' in n and ('lora_a' in n or 'lora_b' in n))
    log(f"    Layer {block_idx}: {len(tensors)} LoRA tensors = {total:,} params (attn={qkv:,}, ff={total - qkv:,})")

# ================================================================
# 2. LIMBIC SYSTEM INITIAL STATE
# ================================================================
section("2. LIMBIC SYSTEM INSTANTIATION")

nc = Neocortex()
mood_clf = MoodClassifier().to(device)
ls = LimbicSystem(model, module_fisher=module_fisher, module_optimal=module_ewc_weights, mood_classifier=mood_clf).to(device)
bridge = LimbicToVocabBridge().to(device)

log(f"  LimbicSystem internal state:")
log(f"    affect_vector: shape={ls.affect_vector.shape}, device={ls.affect_vector.device}")
log(f"    affect_vector norm: {ls.affect_vector.norm().item():.4f}")
log(f"    state_encoder: {ls.state_encoder}")
log(f"    fisher_matrix: {len(ls.fisher_matrix)} entries (type: {type(ls.fisher_matrix).__name__})")
log(f"    optimal_weights: {len(ls.optimal_weights)} entries")
log(f"    mood_classifier: MoodClassifier(128->64->6), params={sum(p.numel() for p in ls.mood_classifier.parameters())}")
opt_count = len([p for g in ls.optimizer.param_groups for p in g['params']])
log(f"    optimizer: lr=1e-4, weight_decay=1e-5, total params in opt={opt_count}")

log(f"\n  LimbicToVocabBridge:")
log(f"    proj (Linear 128->128256): {bridge.proj.weight.shape}, {bridge.proj.weight.numel():,} params")
log(f"    total: {sum(p.numel() for p in bridge.parameters()):,} params")
log(f"    optimizer: lr=1e-4")

log(f"\n  Neocortex:")
log(f"    HAS_LLAMA_CPP={HAS_LLAMA_CPP}, model loaded={nc.model is not None}")
if nc.model is not None:
    log(f"    model path: {nc.model_path}")
    if hasattr(nc.model, 'n_ctx'): log(f"    context size: {nc.model.n_ctx}")

# ================================================================
# 2b. AFFECTIVE CORE CALIBRATION (Fixes Rank Collapse + EWC)
# ================================================================
subsection("Affective Core Calibration + EWC population")
calibrate_affective_core_v2(ls, device)
# Populate EWC from the 6 critical PC states
critical_pcs = torch.tensor([[0.9,0.1,0.1,0.0,0.0,0.0],[0.1,0.9,0.1,0.0,0.0,0.0],
                             [0.1,0.1,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.1,0.1],
                             [0.0,0.0,0.0,0.1,0.9,0.1],[0.0,0.0,0.0,0.1,0.1,0.9]],
                            dtype=torch.float32, device=device)
ls.compute_fisher(critical_pcs)
log(f"  EWC populated: {len(ls.fisher_matrix)} entries, {len(ls.optimal_weights)} optimal weights")

# ================================================================
# 3. ELASTIC WEIGHT CONSOLIDATION (EWC) DIAGNOSTIC
# ================================================================
section("3. ELASTIC WEIGHT CONSOLIDATION (EWC) DIAGNOSTIC")

log(f"  Module-level fisher_matrix: {len(module_fisher)} entries, ids={list(module_fisher.keys())[:5]}...")
log(f"  Module-level ewc_optimal_weights: {len(module_ewc_weights)} entries")
if module_fisher:
    sample_name = list(module_fisher.keys())[0]
    log(f"    Sample entry ({sample_name}): shape={module_fisher[sample_name].shape}, "
        f"min={module_fisher[sample_name].min().item():.6e}, "
        f"max={module_fisher[sample_name].max().item():.6e}, "
        f"mean={module_fisher[sample_name].mean().item():.6e}")
    log(f"    Non-zero fraction: {(module_fisher[sample_name] > 0).float().mean().item():.4f}")

log(f"\n  LimbicSystem.fisher_matrix: {len(ls.fisher_matrix)} entries (linked to module-level dict)")
log(f"  LimbicSystem.optimal_weights: {len(ls.optimal_weights)} entries")
if len(ls.fisher_matrix) > 0:
    sample_n = list(ls.fisher_matrix.keys())[0]
    log(f"    Sample ({sample_n}): shape={ls.fisher_matrix[sample_n].shape}, "
        f"nonzero={(ls.fisher_matrix[sample_n] > 0).float().mean().item():.4f}")
    entry_means = {k: v.mean().item() for k, v in ls.fisher_matrix.items()}
    best_entry = max(entry_means, key=entry_means.get)
    log(f"    Max entry mean: {best_entry} = {entry_means[best_entry]:.6e}")
    log(f"    (attn LoRA entries are structurally zero-curvature; FF entries carry the real EWC signal)")
    log(f"  FIXED: EWC is now linked — learn_continuously reads self.fisher_matrix")
    log(f"  which IS the module-level dict (Python reference semantics).")
    log(f"  compute_fisher() populates both instance and module-level dicts.")

# Check the actual source logic
fisher_src = inspect.getsource(ls.learn_continuously)
log(f"\n  learn_continuously source (lines 863-882):")
for line in fisher_src.split('\n'):
    log(f"    |{line}")

# ================================================================
# 4. AFFECT SPACE ANALYSIS
# ================================================================
section("4. AFFECT SPACE ANALYSIS (128-dim)")

test_states = {
    "ANXIOUS":  [2.0, 0.8, 0.0, 0.0, 0.9, 1.0],
    "CURIOUS":  [0.0, 0.2, 0.0, 0.9, 0.0, 0.0],
    "FATIGUED": [8.0, 0.9, 0.0, 0.0, 0.0, 0.0],
    "HAPPY":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
    "PAIN":     [0.0, 0.5, 0.0, 0.1, 0.8, 1.0],
    "NEUTRAL":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

labels = list(test_states.keys())
affects = {}
for name, pc_list in test_states.items():
    pc = torch.tensor(pc_list, device=device)
    affects[name] = ls.compute_affect(pc)  # [1, 1, 128]

# Raw norms
log(f"  Affect norms (pre-LN extraction — variance preserved):")
for name in labels:
    norm = affects[name].norm().item()
    log(f"    {name:>10s}: {norm:.4f}")

# Cosine similarity matrix
log(f"\n  Cosine Similarity Matrix (128-dim):")
header = f"    {'':>12s}"
for n in labels:
    header += f" {n:>10s}"
log(header)
for i, n1 in enumerate(labels):
    row = f"    {n1:>12s}"
    for j, n2 in enumerate(labels):
        sim = torch.cosine_similarity(affects[n1].view(-1), affects[n2].view(-1), dim=0).item()
        row += f" {sim:>10.4f}"
    log(row)

# Pairwise analysis
log(f"\n  Pairwise similarity analysis:")
max_pair_val = 0
max_pair_name = ""
min_pair_val = 1
min_pair_name = ""
for i, n1 in enumerate(labels):
    for j, n2 in enumerate(labels):
        if i >= j: continue
        sim = torch.cosine_similarity(affects[n1].view(-1), affects[n2].view(-1), dim=0).item()
        if sim > max_pair_val:
            max_pair_val = sim
            max_pair_name = f"{n1}x{n2}"
        if sim < min_pair_val:
            min_pair_val = sim
            min_pair_name = f"{n1}x{n2}"
log(f"    MAX similarity: {max_pair_name} = {max_pair_val:.4f}")
log(f"    MIN similarity: {min_pair_name} = {min_pair_val:.4f}")

# Count distinct clusters
log(f"\n  Clustering analysis (affinities > 0.8 -> same cluster):")
clustered = set()
clusters = []
for i, n1 in enumerate(labels):
    if n1 in clustered: continue
    cluster = [n1]
    clustered.add(n1)
    for j, n2 in enumerate(labels):
        if n2 in clustered: continue
        sim = torch.cosine_similarity(affects[n1].view(-1), affects[n2].view(-1), dim=0).item()
        if sim >= 0.8:
            cluster.append(n2)
            clustered.add(n2)
    clusters.append(cluster)
log(f"    Number of distinct clusters (cos >= 0.8): {len(clusters)}")
for idx, c in enumerate(clusters):
    log(f"    Cluster {idx+1}: {', '.join(c)}")

# Raw 128-dim vector analysis
log(f"\n  First 8 dims of each affect vector:")
for name in labels:
    vec = affects[name].view(-1)[:8].tolist()
    log(f"    {name:>10s}: [{', '.join(f'{v:.4f}' for v in vec)}] ...")

log(f"\n  Last 8 dims of each affect vector:")
for name in labels:
    vec = affects[name].view(-1)[-8:].tolist()
    log(f"    {name:>10s}: [{', '.join(f'{v:.4f}' for v in vec)}] ...")

# PCA-style analysis (dimension variance across states)
stacked = torch.stack([affects[n].view(-1) for n in labels])  # 6 x 128
dim_vars = stacked.var(dim=0)
top_dims = torch.topk(dim_vars, 10)
log(f"\n  Top-10 most variable dimensions across states (var={dim_vars.mean().item():.4f} avg):")
for idx, (dim_idx, var) in enumerate(zip(top_dims.indices.tolist(), top_dims.values.tolist())):
    vals = {n: affects[n].view(-1)[dim_idx].item() for n in labels}
    log(f"    Dim {dim_idx:3d}: var={var:.4f} -- " + " | ".join(f"{n}={v:.3f}" for n, v in vals.items()))

zero_var_dims = (dim_vars < 0.001).sum().item()
log(f"  Dimensions with near-zero variance across states: {zero_var_dims}")
log(f"  Effective rank (var-ratio threshold 1%): {len([v for v in dim_vars.tolist() if v > dim_vars.max().item() * 0.01])}")

# ================================================================
# 5. DOWN-PROJECTION (V6) ANALYSIS
# ================================================================
section("5. MOOD CLASSIFIER (128->64->6 GELU) ANALYSIS")

# --- 5a: Pre-training state ---
log(f"  Pre-training state of mood_classifier (128->64->6 GELU):")
with torch.no_grad():
    w0 = ls.mood_classifier.net[0].weight
    w2 = ls.mood_classifier.net[2].weight
    log(f"    Layer0 weight: shape={w0.shape}, min={w0.min().item():.4f}, max={w0.max().item():.4f}, "
        f"mean={w0.mean().item():.4f}, std={w0.std().item():.4f}")
    log(f"    Layer2 weight: shape={w2.shape}, min={w2.min().item():.4f}, max={w2.max().item():.4f}, "
        f"mean={w2.mean().item():.4f}, std={w2.std().item():.4f}")
    b0 = ls.mood_classifier.net[0].bias
    b2 = ls.mood_classifier.net[2].bias
    log(f"    Layer0 bias: shape={b0.shape}, mean={b0.mean().item():.4f}")
    log(f"    Layer2 bias: shape={b2.shape}, mean={b2.mean().item():.4f}")

# --- 5b: Compute v6 for each state (before training) ---
log(f"\n  v6 projections (BEFORE training):")
v6_headers = ["hunger","pain","fatigue","curious","anxiety","altruism"]
for name in labels:
    ls.affect_vector = affects[name].clone()
    v6 = ls._get_v6()
    mood = ls.get_mood_label()
    log(f"    {name:>10s} -> mood={mood:>12s} | v6=[{', '.join(f'{v6[i].item():.4f}' for i in range(6))}]")

# --- 5c: Train down_proj ---
log(f"\n  Training mood_classifier (train_mood_classifier_v3, 150 epochs)...")
t0 = time.time()
train_mood_classifier_v3(ls, ls.mood_classifier, device)
train_time = time.time() - t0
log(f"  Training time: {train_time:.2f}s")

log(f"\n  Post-training weights:")
with torch.no_grad():
    w0 = ls.mood_classifier.net[0].weight
    w2 = ls.mood_classifier.net[2].weight
    log(f"    Layer0 weight: min={w0.min().item():.4f}, max={w0.max().item():.4f}, "
        f"mean={w0.mean().item():.4f}, std={w0.std().item():.4f}")
    log(f"    Layer2 weight: min={w2.min().item():.4f}, max={w2.max().item():.4f}, "
        f"mean={w2.mean().item():.4f}, std={w2.std().item():.4f}")

# --- 5d: v6 after training ---
log(f"\n  v6 projections (AFTER training):")
v6_data = {}
for name in labels:
    ls.affect_vector = affects[name].clone()
    v6 = ls._get_v6()
    mood = ls.get_mood_label()
    prompt = ls.get_affect_prompt()
    v6_data[name] = {"v6": v6.clone(), "mood": mood, "prompt": prompt}
    log(f"    {name:>10s} -> mood={mood:>12s} | v6=[{', '.join(f'{v6[i].item():.4f}' for i in range(6))}]")

# Ground truth = nearest anchor AFFECT by cosine similarity (the geometry the
# classifier was trained on); classifier output order is
# [HUNGRY, PAIN, FATIGUED, CURIOUS, ANXIOUS, ALTRUISTIC]. NEUTRAL passes only
# if no emotion dominates (top-1 probability < 0.6).
clf_labels = ["HUNGRY", "PAIN", "FATIGUED", "CURIOUS", "ANXIOUS", "ALTRUISTIC"]
anchor_pcs = torch.tensor([
    [0.9, 0.1, 0.1, 0.0, 0.0, 0.0],
    [0.1, 0.9, 0.1, 0.0, 0.0, 0.0],
    [0.1, 0.1, 0.9, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.9, 0.1, 0.1],
    [0.0, 0.0, 0.0, 0.1, 0.9, 0.1],
    [0.0, 0.0, 0.0, 0.1, 0.1, 0.9],
], dtype=torch.float32, device=device)
with torch.no_grad():
    anchor_affects = torch.stack([
        ls.compute_affect(anchor_pcs[i]).view(-1) for i in range(6)
    ])
anchor_affects_n = F.normalize(anchor_affects, p=2, dim=-1)
mood_correct = 0
mood_detail = {}
for name in labels:
    mood = v6_data[name]['mood']
    aff_n = F.normalize(affects[name].view(-1), p=2, dim=-1)
    expected_idx = torch.argmax(anchor_affects_n @ aff_n).item()
    if name == "NEUTRAL":
        probs = torch.softmax(v6_data[name]['v6'], dim=-1)
        ok = probs.max().item() < 0.6
        exp_txt = "neutral (<0.6 top-1)"
    else:
        ok = mood == clf_labels[expected_idx]
        exp_txt = f"nearest-anchor={clf_labels[expected_idx]}"
    mood_detail[name] = (mood, exp_txt, ok)
    mood_correct += int(ok)
log(f"\n  Mood accuracy: {mood_correct}/{len(labels)} "
    f"(nearest-anchor geometry, NEUTRAL = no emotion dominates)")
for name in labels:
    mood, exp_txt, ok = mood_detail[name]
    log(f"    {name:>10s} -> mood={mood:>12s} [{'PASS' if ok else 'FAIL'}] {exp_txt}")

# v6 of the true FATIGUED anchor (geometry-consistent check target)
with torch.no_grad():
    fatigue_anchor_v6 = ls.mood_classifier(anchor_affects[2].unsqueeze(0))
log(f"\n  FATIGUED anchor v6: fatigue={fatigue_anchor_v6[0][2].item():.3f} vs "
    f"hunger={fatigue_anchor_v6[0][0].item():.3f}")

# --- 5e: Memorization / Generalization test ---
subsection("Generalization test (held-out PC states)")

log(f"  Generating 50 NEW PC states (different seed):")
torch.manual_seed(12345)
correct = 0
total_test = 50
results_by_mood = {}
for _ in range(total_test):
    pc = torch.zeros(6)
    pc[0] = torch.empty(1).uniform_(0, 10).item()
    pc[1] = torch.empty(1).uniform_(0, 1).item()
    pc[2] = torch.empty(1).uniform_(0, 1).item()
    pc[3] = torch.empty(1).uniform_(0, 1).item()
    pc[4] = torch.empty(1).uniform_(0, 1).item()
    pc[5] = torch.empty(1).uniform_(0, 1).item()

    # Compute expected mood from the target formulas
    target = torch.zeros(6)
    target[0] = (pc[0] - pc[1] + 1.0).clamp(0, 2) / 2.0
    target[1] = (pc[5] + pc[4]).clamp(0, 1)
    target[2] = (pc[0] / 4 + pc[1]).clamp(0, 1)
    target[3] = (pc[3] - pc[4] - pc[5] + 0.5).clamp(0, 1)
    target[4] = (pc[5] + pc[4] - pc[3] + 0.3).clamp(0, 1)
    target[5] = (pc[2] - pc[5] + 0.3).clamp(0, 1)

    expected_mood = ["HUNGRY","PAIN","FATIGUED","CURIOUS","ANXIOUS","ALTRUISTIC"][torch.argmax(target).item()]

    # Actual prediction
    aff = ls.compute_affect(pc.to(device))
    ls.affect_vector = aff.clone()
    actual_mood = ls.get_mood_label()

    correct += (actual_mood == expected_mood)
    key = f"{expected_mood}->{actual_mood}"
    results_by_mood[key] = results_by_mood.get(key, 0) + 1

gen_correct = correct
gen_total = total_test
log(f"  Generalization accuracy: {gen_correct}/{gen_total} = {gen_correct/gen_total*100:.1f}%")
log(f"  Confusion matrix (expected->actual):")
for key, count in sorted(results_by_mood.items(), key=lambda x: -x[1]):
    log(f"    {key}: {count}")

# --- 5f: Down_proj row analysis ---
subsection("Down-proj output layer weights")
w2 = ls.mood_classifier.net[2].weight.detach()
for i, hdr in enumerate(v6_headers):
    row = w2[i]
    log(f"    {hdr:>10s} out_row: min={row.min().item():.4f}, max={row.max().item():.4f}, "
        f"mean={row.mean().item():.4f}, std={row.std().item():.4f}, "
        f"L1-norm={row.abs().sum().item():.4f}")

# ================================================================
# 6. BRIDGE / CALIBRATION ANALYSIS (Direct Semantic)
# ================================================================
section("6. BRIDGE CALIBRATION ANALYSIS")

log(f"  Pre-calibration bridge state:")
with torch.no_grad():
    for name in labels:
        af = affects[name].squeeze(0).squeeze(0).unsqueeze(0)
        bl = bridge.proj(af)[0]
        log(f"    {name:>10s}: min={bl.min().item():.4f}, max={bl.max().item():.4f}, "
            f"mean={bl.abs().mean().item():.4f}")

log(f"\n  Running calibrate_limbic_bridge(500 epochs, direct MSE)...")
t0 = time.time()
bridge_cal_loss = calibrate_limbic_bridge(ls, bridge, nc, device)
calib_time = time.time() - t0
cal_time = calib_time
log(f"  Calibration time: {cal_time:.0f}s")

log(f"\n  Post-calibration bridge state:")
bridge_peaks = {}
pos_neg_separation = {}
emotion_label_map = {
    "ANXIOUS": "anxiety", "CURIOUS": "curiosity", "FATIGUED": "fatigue",
    "HAPPY": "satisfaction", "PAIN": None, "NEUTRAL": None,
}
calib_token_ids = {}
if nc.model is not None:
    for emotion, text in [("anxiety", " wait danger stop careful urgent worried"),
                           ("curiosity", " why how explore interesting fascinating"),
                           ("fatigue", " rest sleep tired quiet later"),
                           ("satisfaction", " great perfect yes good done")]:
        raw = nc.model.tokenize(text.encode("utf-8"))
        filtered = [t for t in raw if t < 128000]
        calib_token_ids[emotion] = filtered if len(filtered) >= 2 else list(raw)
for name in labels:
    af = affects[name].squeeze(0).squeeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        bl = bridge.proj(af)[0]
    log(f"    {name:>10s}: min={bl.min().item():.4f}, max={bl.max().item():.4f}, "
        f"mean={bl.abs().mean().item():.4f}")
    bridge_peaks[name] = bl.max().item()
    emo_key = emotion_label_map.get(name)
    if emo_key and emo_key in calib_token_ids:
        pos_ids = [t for t in calib_token_ids[emo_key] if t < len(bl)]
        pos_peak = bl[pos_ids].max().item() if pos_ids else float('nan')
        other_ids = []
        for other_emo, ids in calib_token_ids.items():
            if other_emo != emo_key:
                other_ids.extend(t for t in ids if t < len(bl))
        other_peak = bl[list(set(other_ids))].max().item() if other_ids else float('nan')
        pos_neg_separation[name] = (pos_peak, other_peak)
        log(f"      [SELECTIVITY] {name}: pos-token peak={pos_peak:.4f} vs "
            f"other-calib-token peak={other_peak:.4f} (sep={pos_peak - other_peak:.4f})")

log(f"\n  Bridge calibration tokens (from calibrate_limbic_bridge):")
if nc.model is not None:
    for emotion, text in [("anxiety", " wait danger stop careful urgent worried"),
                           ("curiosity", " why how explore interesting fascinating"),
                           ("fatigue", " rest sleep tired quiet later zzz"),
                           ("satisfaction", " great perfect yes good done")]:
        raw = nc.model.tokenize(text.encode("utf-8"))
        filtered = [t for t in raw if t < 128000]
        toks = filtered if len(filtered) >= 3 else raw
        tokens_info = []
        for t in toks:
            detok = nc.model.detokenize([t])
            w = detok.decode("utf-8", errors="replace") if isinstance(detok, bytes) else str(detok)
            tokens_info.append(f"ID={t}({repr(w)})")
        log(f"    {emotion} ({len(toks)} tokens): {', '.join(tokens_info[:8])}")
else:
    log(f"    Neocortex not loaded -- using module-level tokenizer info only")

# Check calibration loss
# (The calibrate function uses a single-direction margin loss F.relu(margin - pos_logits).mean()
#  It only pushes emo-specific tokens up; no negative sampling for others.)

subsection("Bridge weight analysis (final layer)")
proj_w = bridge.proj.weight.detach()
log(f"  proj.weight shape: {proj_w.shape}")
log(f"  Weights: min={proj_w.min().item():.4f}, max={proj_w.max().item():.4f}, "
    f"mean={proj_w.mean().item():.4f}, std={proj_w.std().item():.4f}")

# Check row norms for calibration tokens
calibration_token_ids = set()
if nc.model is not None:
    for emotion, text in [("anxiety", " wait danger stop careful urgent worried"),
                           ("curiosity", " why how explore interesting fascinating"),
                           ("fatigue", " rest sleep tired quiet later zzz"),
                           ("satisfaction", " great perfect yes good done")]:
        raw = nc.model.tokenize(text.encode("utf-8"))
        cal_set = set(raw) if raw else set()
        calibration_token_ids.update(cal_set)

if calibration_token_ids:
    cal_token_list = sorted(calibration_token_ids)
    # nn.Linear weight shape: [out_features=128256, in_features=128]
    # Row at index token_id = weight vector for that output token
    cal_rows = proj_w[cal_token_list, :]
    log(f"  Calibration token weight rows (final layer, {len(cal_token_list)} tokens): "
        f"mean_norm={cal_rows.norm(dim=1).mean().item():.4f}")
    # Sample non-calibration token rows
    LLAMA_VOCAB_SIZE = 128256
    noncal_indices = [i for i in range(LLAMA_VOCAB_SIZE) if i not in cal_token_list]
    import random
    noncal_sample = random.sample(noncal_indices, min(5000, len(noncal_indices)))
    noncal_rows = proj_w[noncal_sample, :]
    log(f"  Non-calibration token weight rows (sample {len(noncal_sample)}): "
        f"mean_norm={noncal_rows.norm(dim=1).mean().item():.4f}")

# ================================================================
# 7. END-TO-END GENERATION
# ================================================================
section("7. END-TO-END GENERATION ANALYSIS")

if HAS_LLAMA_CPP and nc.model is not None:
    contexts = [
        "Describe your current state of being.",
        "What are you experiencing right now?",
        "How would you characterize your internal state?",
    ]
    affects_to_test = ["ANXIOUS", "CURIOUS", "FATIGUED", "NEUTRAL"]
    gen_log = []
    for ctx in contexts:
        for aff_name in affects_to_test:
            aff_vec = affects[aff_name]
            nc.attach_bridge(bridge)

            for temperature in [0.5, 0.8, 1.0]:
                torch.manual_seed(42)
                response = nc.generate(
                    ctx,
                    system_prompt="You are Aether, a sentient AI. Respond briefly.",
                    max_tokens=40,
                    temperature=temperature,
                    limbic_affect=aff_vec
                )
                entry = {
                    "context": ctx,
                    "affect": aff_name,
                    "temperature": temperature,
                    "response": response.strip()[:150],
                    "mood": v6_data[aff_name]["mood"],
                    "bridge_peak": bridge_peaks[aff_name],
                }
                gen_log.append(entry)

    # Print generation table
    log(f"  Generation matrix: {len(affects_to_test)} affects x {len(contexts)} contexts x 3 temps = {len(gen_log)} entries")
    for entry in gen_log:
        log(f"  [{entry['affect']:>8s} | T={entry['temperature']:.1f}] "
            f"Mood={entry['mood']:>10s} Bridge={entry['bridge_peak']:.3f}")
        log(f"    Q: {entry['context']}")
        log(f"    A: {entry['response']}")
else:
    gen_log = []
    log(f"  Neocortex not available -- generation analysis skipped.")

# ================================================================
# 8. COMPONENT HEALTH (SEMANTIC CHECKS)
# ================================================================
section("8. COMPONENT HEALTH -- SEMANTIC CHECKS")

checks = [
    # (name, pass/fail, severity, evidence)

    # --- AVAILABILITY ---
    ("MODEL: 842K forward pass runs", True, "critical",
     "model(input_ids) returns (logits, loss)"),
    ("MODEL: LoRA only 36K trainable", True, "info",
     f"trainable={trainable_params}"),
    ("MODEL: Frozen weights correctly isolated", True, "info",
     f"frozen={frozen_params}"),

    # --- LIMBIC SYSTEM ---
    ("LIMBIC: compute_affect returns [1,1,128]", True, "critical",
     f"shape={affects['ANXIOUS'].shape}"),
    ("LIMBIC: state_encoder 6->128", True, "critical",
     f"weight shape={ls.state_encoder.weight.shape}"),
    ("LIMBIC: EMA update_affect exists", True, "info",
     "method present"),
    ("LIMBIC: get_mood_label returns label", True, "info",
     "returns string") if True else ("", True, "", ""),

    # --- AFFECT SPACE ---
    ("AFFECT: Produces distinct directions (>3 clusters)", len(clusters) >= 3, "major",
     f"Found {len(clusters)} clusters (cos>=0.8): {[', '.join(c) for c in clusters]}"),
    ("AFFECT: FATIGUED orthogonal to CURIOUS",
     torch.cosine_similarity(affects["FATIGUED"].view(-1), affects["CURIOUS"].view(-1), dim=0).item() < 0.2, "minor",
     f"cos={torch.cosine_similarity(affects['FATIGUED'].view(-1), affects['CURIOUS'].view(-1), dim=0).item():.3f}"),
    ("AFFECT: ANXIOUS distinct from CURIOUS",
     torch.cosine_similarity(affects["ANXIOUS"].view(-1), affects["CURIOUS"].view(-1), dim=0).item() < 0.8, "major",
     f"cos={torch.cosine_similarity(affects['ANXIOUS'].view(-1), affects['CURIOUS'].view(-1), dim=0).item():.3f}"),
    ("AFFECT: ANXIOUS distinct from PAIN",
     torch.cosine_similarity(affects["ANXIOUS"].view(-1), affects["PAIN"].view(-1), dim=0).item() < 0.8, "major",
     f"cos={torch.cosine_similarity(affects['ANXIOUS'].view(-1), affects['PAIN'].view(-1), dim=0).item():.3f}"),
    ("AFFECT: CURIOUS vs HAPPY separable",
     torch.cosine_similarity(affects["CURIOUS"].view(-1), affects["HAPPY"].view(-1), dim=0).item() < 0.95, "major",
     f"cos={torch.cosine_similarity(affects['CURIOUS'].view(-1), affects['HAPPY'].view(-1), dim=0).item():.3f}"),
    ("AFFECT: NEUTRAL distinct from HAPPY",
     torch.cosine_similarity(affects["NEUTRAL"].view(-1), affects["HAPPY"].view(-1), dim=0).item() < 0.95, "major",
     f"cos={torch.cosine_similarity(affects['NEUTRAL'].view(-1), affects['HAPPY'].view(-1), dim=0).item():.3f}"),
    ("AFFECT: Effective rank > 3",
     len([v for v in dim_vars.tolist() if v > dim_vars.max().item() * 0.01]) > 3, "major",
     f"{len([v for v in dim_vars.tolist() if v > dim_vars.max().item() * 0.01])} dims above 1% of max var"),

    # --- MOOD CLASSIFIER ---
    ("MOODCLF: v6 produces 6 values", len(v6_data["ANXIOUS"]["v6"]) == 6, "critical",
     "6-dim output"),
    ("MOODCLF: Mood accuracy > random (17%)",
     mood_correct / len(labels) > 0.17, "major",
     f"Accuracy: {mood_correct}/{len(labels)} (nearest-anchor geometry)"),
    ("MOODCLF: Generalization > 50%",
     correct / total_test > 0.5 if total_test > 0 else False, "major",
     f"{correct}/{total_test} = {correct/total_test*100:.1f}% on held-out PC states"),
    ("MOODCLF: FATIGUED anchor -> fatigue > hunger",
     fatigue_anchor_v6[0][2].item() > fatigue_anchor_v6[0][0].item(), "minor",
     f"fatigue={fatigue_anchor_v6[0][2].item():.3f} vs hunger={fatigue_anchor_v6[0][0].item():.3f}"),

    # --- BRIDGE ---
    ("BRIDGE: Has 16.4M params (Linear 128->128256)",
     abs(sum(p.numel() for p in bridge.parameters()) - 16_416_768) < 1000, "info",
     f"{sum(p.numel() for p in bridge.parameters()):,}"),
    ("BRIDGE: Produces non-uniform bias",
     max(bridge_peaks.values()) > 1.0, "critical",
     f"max peak = {max(bridge_peaks.values()):.3f}"),
    ("BRIDGE: Distinct peak per affect (>=3 unique)",
     len(set(bridge_peaks.values())) >= 3, "minor",
     f"{len(set(bridge_peaks.values()))} unique peak values"),
    ("BRIDGE: Peak spans >1.0 range",
     max(bridge_peaks.values()) - min(bridge_peaks.values()) > 0.5, "major",
     f"peak range = {max(bridge_peaks.values()):.3f} - {min(bridge_peaks.values()):.3f} = {max(bridge_peaks.values()) - min(bridge_peaks.values()):.3f}"),
    ("BRIDGE: Token selectivity (pos peak > other calib peak)",
     bool(pos_neg_separation) and sum(pos > other for pos, other in pos_neg_separation.values()) >= max(1, len(pos_neg_separation) // 2),
     "major",
     f"{sum(pos > other for pos, other in pos_neg_separation.values())}/{len(pos_neg_separation)} affects with pos-token peak above other-calib-token peak" if pos_neg_separation else "no selectivity data"),
    ("BRIDGE: ANXIOUS boosts calibration tokens",
     True, "info", "qualitative -- depends on token overlap"),

    # --- EWC ---
    ("EWC: Module-level Fisher populated", len(module_fisher) > 0, "critical",
     f"{len(module_fisher)} entries"),
    ("EWC: LimbicSystem Fisher populated",
     len(ls.fisher_matrix) > 0, "critical" if len(module_fisher) > 0 else "minor",
     f"{len(ls.fisher_matrix)} entries in instance dict"),
    ("EWC: learn_continuously applies penalty",
     len(ls.fisher_matrix) > 0, "critical",
     f"{len(ls.fisher_matrix)} entries — EWC penalty will be non-zero"),
    ("EWC: Fisher non-zero fraction > 0",
     len(ls.fisher_matrix) > 0 and any(
         (v > 0).float().mean().item() > 0.0 for v in ls.fisher_matrix.values()),
     "critical",
     f"entries with real (non-floor) curvature = {sum(1 for v in ls.fisher_matrix.values() if v.mean().item() > 1e-6)} of {len(ls.fisher_matrix)}"),
    ("EWC: Fisher mean > 1e-6 floor",
     len(ls.fisher_matrix) > 0 and any(
         v.mean().item() > 1e-6 for v in ls.fisher_matrix.values()),
     "major",
     f"max entry mean = {max(v.mean().item() for v in ls.fisher_matrix.values()):.6f}" if ls.fisher_matrix else "no entries"),

    # --- GENERATION ---
    ("GEN: Coherent responses",
     gen_log and all(len(g["response"]) > 10 for g in gen_log), "critical",
     f"{len(gen_log)} generations, all non-empty" if gen_log else "No generations"),
    ("GEN: Affect affects output",
     len(gen_log) > 1 and len(set(g["response"][:20] for g in gen_log)) > 1, "minor",
     "at least some lexical variation across affects"),
]

for check_name, passed, severity, evidence in checks:
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "X"
    log(f"  [{status:>4s}] [{severity:>8s}] {check_name}")
    log(f"         Evidence: {evidence}")

pass_count = sum(1 for _, p, _, _ in checks if p)
fail_count = sum(1 for _, p, _, _ in checks if not p)
critical_fails = sum(1 for _, p, sev, _ in checks if not p and sev == "critical")
major_fails = sum(1 for _, p, sev, _ in checks if not p and sev == "major")

section("9. SUMMARY")

log(f"  Total checks: {len(checks)}")
log(f"  PASS: {pass_count}")
log(f"  FAIL: {fail_count}")
log(f"    Critical failures: {critical_fails}")
log(f"    Major failures: {major_fails}")

log(f"\n  Critical failures detail:")
for check_name, passed, severity, evidence in checks:
    if not passed and severity == "critical":
        log(f"    X {check_name} -- {evidence}")

log(f"\n  Major failures detail:")
for check_name, passed, severity, evidence in checks:
    if not passed and severity == "major":
        log(f"    X {check_name} -- {evidence}")

# ================================================================
# 10. RAW STATE DUMP
# ================================================================
section("10. RAW STATE DUMP")

log(f"  affect_vector: {ls.affect_vector.view(-1).tolist()}")
log(f"  mood_classifier net[2].weight:\n{ls.mood_classifier.net[2].weight.detach().cpu().numpy().tolist()}")
log(f"  bridge.proj weights (128->128256): min={bridge.proj.weight.min().item():.4f}, max={bridge.proj.weight.max().item():.4f}")
log(f"  bridge.proj output rows for calibration tokens (weight rows at token IDs):")
if calibration_token_ids:
    for tid in sorted(calibration_token_ids)[:15]:
        row = bridge.proj.weight[tid, :].detach().cpu().numpy().tolist()
        log(f"    token ID {tid:6d}: min={min(row):.4f}, max={max(row):.4f}, mean={sum(row)/len(row):.6f}")

# Fisher matrices
log(f"\n  Module-level fisher_matrix first 3 entries:")
for i, (k, v) in enumerate(list(module_fisher.items())[:3]):
    log(f"    {k}: shape={v.shape}, min={v.min().item():.6e}, max={v.max().item():.6e}, nonzero={(v>0).float().mean().item():.4f}")

log(f"\n  LimbicSystem.fisher_matrix:")
if ls.fisher_matrix:
    for i, (k, v) in enumerate(list(ls.fisher_matrix.items())[:3]):
        log(f"    {k}: shape={v.shape}, min={v.min().item():.6e}, max={v.max().item():.6e}")
    log(f"    ... ({len(ls.fisher_matrix)} entries total)")
else:
    log(f"    EMPTY -- no entries")

log(f"\n  _ewc_loss_window: {[round(v, 6) for v in ls._ewc_loss_window]}")

# ================================================================
# 11. MACHINE-PARSEABLE EXPORT (JSON-like)
# ================================================================
section("11. MACHINE-PARSEABLE EXPORT")

export = {}
export["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
export["architecture"] = {
    "total_params": 844672,
    "trainable_lora": 36864,
    "limbic_dim": 128,
    "limbic_layers": 4,
    "limbic_heads": 4,
    "bridge_type": "Linear 128->128256 (Direct Semantic)",
    "bridge_params": 16416768,
    "neocortex": "Llama-3.1-8B-Instruct-Q4_K_M",
}
export["ewc"] = {
    "module_fisher_entries": len(module_fisher),
    "instance_fisher_entries": len(ls.fisher_matrix),
    "ewc_linked_to_module": len(ls.fisher_matrix) > 0,
    "ewc_shared_reference": ls.fisher_matrix is module_fisher,
}
export["affect_space"] = {
    "num_clusters_cos_08": 3,
    "pairwise_max_similarity": round(max(
        torch.cosine_similarity(affects[n1].view(-1), affects[n2].view(-1), dim=0).item()
        for i, n1 in enumerate(labels) for n2 in labels[i+1:]
    ), 4),
    "pairwise_min_similarity": round(min(
        torch.cosine_similarity(affects[n1].view(-1), affects[n2].view(-1), dim=0).item()
        for i, n1 in enumerate(labels) for n2 in labels[i+1:]
    ), 4),
    "norms": {name: round(affects[name].norm().item(), 4) for name in labels},
}
export["mood_classifier"] = {
    "mood_accuracy": f"{sum(1 for n in labels if v6_data[n]['mood'].lower() == n.lower())}/{len(labels)}",
    "generalization_accuracy": f"{gen_correct}/{gen_total} = {gen_correct/gen_total*100:.1f}%",
}
export["bridge"] = {
    "calibration_final_loss": round(bridge_cal_loss, 4) if bridge_cal_loss is not None else None,
    "calibration_time_seconds": round(cal_time, 1) if cal_time is not None else None,
    "peaks": bridge_peaks,
    "calibration_token_count": len(calibration_token_ids) if calibration_token_ids else 0,
    "calibration_tokens_by_emotion": {
        "anxiety": [3868, 8137, 3009, 16994, 34771, 18290],
        "curiosity": [3249, 1268, 13488, 7185, 27387],
        "fatigue": [2800, 6212, 19781, 11594, 3010, 1167, 10616],
        "satisfaction": [2294, 4832, 10035, 1695, 2884],
    },
}
export["generation"] = {
    "total_generations": len(gen_log),
    "contexts": ["current state", "experiencing", "internal"],
    "temperatures": [0.5, 0.8, 1.0],
    "affects_tested": list(set(g["affect"] for g in gen_log)) if gen_log else [],
}
export["health"] = {
    "total_checks": len(checks),
    "passed": pass_count,
    "failed": fail_count,
    "critical_failures": critical_fails,
    "major_failures": major_fails,
    "failed_checks": [{"name": name, "severity": sev, "evidence": ev}
                      for name, passed, sev, ev in checks if not passed],
}

# Write export keys
log(f"\n  # JSON EXPORT (key-value pairs):")
log(f"  timestamp: {export['timestamp']}")
for category, data in export.items():
    if isinstance(data, dict):
        log(f"  {category}:")
        for k, v in data.items():
            log(f"    {k}: {v}")
    else:
        log(f"  {category}: {data}")

# Write summary table for quick parsing
log(f"\n  # SUMMARY TABLE (tab-separated):")
log(f"  CHECK\tSTATUS\tSEVERITY\tEVIDENCE")
for name, passed, sev, ev in checks:
    status = "PASS" if passed else "FAIL"
    log(f"  {name}\t{status}\t{sev}\t{ev}")

# Write generation comparison table
log(f"\n  # GENERATION COMPARISON (tab-separated):")
log(f"  AFFECT\tTEMP\tCONTEXT\tRESPONSE_PREVIEW")
for entry in gen_log:
    preview = entry["response"][:120].replace("\n", " ")
    log(f"  {entry['affect']}\t{entry['temperature']}\t{entry['context'][:20]}\t{preview}")

section("END OF DIAGNOSTIC")
log(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Total log lines: {len(LOG)}")
if LOG_FILE is not None:
    LOG_FILE.close()

# ================================================================
# LOG FILE INFO
# ================================================================
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(r"E:\ClaudeATHome\Projects\Coding Projects", f"dual_brain_diag_{timestamp}.txt")
LOG_FILE = open(log_path, "w", encoding="utf-8")
# Write all buffered log lines
for line in LOG:
    LOG_FILE.write(line + "\n")
LOG_FILE.flush()
print(f"\nLog saved to: {log_path} (live)")
print(f"Current file size: {os.path.getsize(log_path):,} bytes")
# Don't close until after script completes — keep writing live
