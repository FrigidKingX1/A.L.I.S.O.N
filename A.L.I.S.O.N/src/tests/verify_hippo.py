import time, torch, torch.nn.functional as F
import alison_core as ica

print("memory_index type:", type(ica.memory_index).__name__)
print("active_inference type:", type(ica.active_inference).__name__)
print("active_inference device:", next(ica.active_inference.parameters()).device)

# --- Active Inference gamma behavior ---
aic = ica.active_inference
zero = torch.zeros(1, 128)
big = torch.ones(1, 128) * 5.0
g_floor = aic.update_precision(zero, zero).item()      # identical states -> low FE -> ~floor
g_high = aic.update_precision(zero, big).item()        # huge error -> high FE -> ~ceil
print(f"gamma(zero error)={g_floor:.3f}  gamma(huge error)={g_high:.3f}")
assert g_floor < g_high, "gamma must rise with prediction error"
assert 0.1 <= g_floor <= 2.0 and 0.1 <= g_high <= 2.0

# --- Hippocampal fused recall ---
idx = ica.memory_index
q = torch.randn(1, 128)
fused = idx.forward(q)
print("forward() shape:", tuple(fused.shape), "is_bipolar:", bool((fused.abs() == 1).all()))
assert fused.shape == (10000,) and (fused.abs() == 1).all()

idx.store(q, "alpha-mem", 1.0)
idx.store(torch.randn(1, 128), "beta-mem", 0.0)
res = idx.recall(q, k=2)
print("recall() top-2:", [(t[:12], round(v, 3), round(s, 3)) for t, v, s in res])
assert len(res) == 2 and res[0][0] == "alpha-mem" and res[0][2] > 0.9, "fused recall should rank the stored query highest"
print("UNIT CHECKS PASSED")
