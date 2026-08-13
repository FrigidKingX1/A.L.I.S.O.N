"""W1 verification: Larimar selective unbinding on the real HippocampalMemoryIndex.

Asserts what the code actually does (not the spec's equations):
  * forget_pattern() reduces the Fast Weight matrix M_t norm,
  * it changes the fast-weight recall of the forgotten key,
  * recall() no longer returns the scrubbed entry,
  * sibling memories are left intact.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alison_core import HippocampalMemoryIndex


def make_key(seed):
    torch.manual_seed(seed)
    return torch.randn(128)


def main():
    idx = HippocampalMemoryIndex(d_model=128, vsa_dim=10000, capacity=50)
    k1 = make_key(1)
    k2 = make_key(2)
    idx.store(k1, "sensitive_A", valence=0.5)
    idx.store(k2, "memory_B", valence=0.2)

    before = idx.recall(k1, k=3)
    before_top = before[0][0] if before else None
    norm_before = idx.M_t.norm().item()
    fwp_before = idx.retrieve_fast_weight(k1).norm().item()

    idx.forget_pattern(k1, erasure_rate=0.5)

    norm_after = idx.M_t.norm().item()
    fwp_after = idx.retrieve_fast_weight(k1).norm().item()
    after_texts = [t for t, _, _ in idx.recall(k1, k=3)]

    ok = True
    if not (norm_after < norm_before):
        print("FAIL: M_t norm did not decrease"); ok = False
    if abs(fwp_after - fwp_before) < 1e-6:
        print("FAIL: fast-weight recall unchanged after forget"); ok = False
    if "sensitive_A" in after_texts:
        print("FAIL: sensitive_A still recalled after forget_pattern"); ok = False
    if before_top != "sensitive_A":
        print(f"WARN: baseline top was {before_top!r}, expected 'sensitive_A'")
    if "memory_B" not in [t for t, _, _ in idx.recall(k2, k=3)]:
        print("FAIL: sibling memory_B lost after forgetting A"); ok = False

    if ok:
        print("LARIMAR UNBINDING OK: M_t %.1f -> %.1f | fwp %.1f -> %.1f | "
              "recall degraded, sibling intact" % (
                  norm_before, norm_after, fwp_before, fwp_after))
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
