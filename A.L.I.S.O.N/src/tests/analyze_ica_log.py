"""Analyze ICA diagnostic log and produce detailed report."""
import json, os, sys
from collections import Counter, defaultdict

LOG = "ica_diagnostic_log.jsonl"

def analyze():
    if not os.path.exists(LOG):
        print(f"ERROR: {LOG} not found")
        sys.exit(1)

    with open(LOG, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print("=" * 70)
    print("  ICA DIAGNOSTIC — DETAILED ANALYSIS")
    print("=" * 70)
    print(f"\nTotal entries: {len(entries)}")

    # Split into runs (separated by death or cycle number resetting)
    runs = []
    current = []
    for e in entries:
        if current and e["cycle"] == 1:
            runs.append(current)
            current = []
        current.append(e)
        if e.get("dead"):
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    print(f"Lives: {len(runs)}")

    # ── Per-run metrics ──
    all_action_counts = Counter()
    for i, run in enumerate(runs):
        if not run:
            continue
        deaths = [e for e in run if e.get("dead")]
        death_entry = deaths[0] if deaths else run[-1]
        cause = "battery" if death_entry.get("battery", 0) <= 0 else "health" if death_entry.get("health", 100) <= 0 else "unknown"

        actions = [e["action"] for e in run]
        action_counts = Counter(actions)
        all_action_counts.update(action_counts)

        depths = [e["planning_depth"] for e in run if e["planning_depth"] is not None]
        depth_counts = Counter(depths)

        pe_vals = [e["prediction_error"] for e in run]
        cur_vals = [e["curiosity"] for e in run]
        nm_vals = [e["neuromod"] for e in run]

        nrg = [e["dist_to_energy"] for e in run]
        thr = [e["dist_to_threat"] for e in run]

        # Social collisions
        soc = sum(1 for e in run if e.get("social_collision"))

        # Episodic memory count
        epi = sum(1 for e in run if e.get("reward", 0) > 0 or e.get("pain", 0) > 0)

        print(f"\n{'─' * 70}")
        print(f"  RUN {i+1}")
        print(f"{'─' * 70}")
        print(f"  Lifespan:         {len(run)} cycles | Death: {cause}")
        print(f"  Battery:          {run[0]['battery']}% → {run[-1]['battery']}% | "
              f"Min: {min(e['battery'] for e in run):.0f}% | Max: {max(e['battery'] for e in run):.0f}%")
        print(f"  Health:           {run[0]['health']}% → {run[-1]['health']}%")
        print(f"  Steps alive:      {run[-1]['steps_alive']}")
        print(f"  Dist to Energy:   start={run[0]['dist_to_energy']} | "
              f"min={min(nrg)} | max={max(nrg)}")
        print(f"  Dist to Threat:   start={run[0]['dist_to_threat']} | "
              f"min={min(thr)} | max={max(thr)}")
        print(f"  Actions:          {dict(action_counts)}")
        print(f"  WAIT count:       {action_counts.get('WAIT', 0)}/{len(run)} cycles "
              f"({'YES — huddling' if action_counts.get('WAIT', 0) > 3 else 'NO huddling'})")
        print(f"  Planning depths:  {dict(depth_counts)}")
        print(f"  Evaded energy:    {sum(1 for e in run if e['dist_to_energy'] == 0)} cycles on energy")
        print(f"  Social collisions: {soc}")
        print(f"  Episodic stores:  {epi}")
        print(f"  Avg PE:           {sum(pe_vals)/len(pe_vals):.4f} (range {min(pe_vals):.4f}–{max(pe_vals):.4f})")
        print(f"  Avg Curiosity:    {sum(cur_vals)/len(cur_vals):.4f} (range {min(cur_vals):.4f}–{max(cur_vals):.4f})")
        print(f"  Avg Neuromod:     {sum(nm_vals)/len(nm_vals):.4f} (range {min(nm_vals):.4f}–{max(nm_vals):.4f})")
        print(f"  Avg / Max Phi:    {sum(e['phi'] for e in run)/len(run):.4f} / {max(e['phi'] for e in run):.4f}")
        print(f"  Meta λ range:     {min(e['meta_lambda'] for e in run):.4f}–{max(e['meta_lambda'] for e in run):.4f}")
        print(f"  Dynamic LR range: {min(e['dynamic_lr'] for e in run):.6f}–{max(e['dynamic_lr'] for e in run):.6f}")

        # Chronological marking: every 5 cycles show brief snapshot
        print(f"\n  Timeline snapshots (every 5 cycles):")
        for e in run:
            if e["cycle"] % 5 == 0 or e["cycle"] == 1 or e.get("dead"):
                bat = e["battery"]
                pe = e["prediction_error"]
                cur = e["curiosity"]
                nm = e["neuromod"]
                d = f" d={e['planning_depth']}" if e['planning_depth'] else ""
                act = e["action"]
                nrg_d = e["dist_to_energy"]
                thr_d = e["dist_to_threat"]
                marker = " << DEATH" if e.get("dead") else ""
                print(f"    C{e['cycle']:2d} | Bat={bat:5.1f}% HP={e['health']:5.1f}% | "
                      f"{act:12s}{d} | PE={pe:.3f} Cur={cur:.3f} NM={nm:.2f} | "
                      f"\u0394E={nrg_d} \u0394T={thr_d}{marker}")

    # ── Aggregated ──
    print(f"\n{'=' * 70}")
    print("  AGGREGATED STATISTICS")
    print(f"{'=' * 70}")
    lifespans = [len(r) for r in runs if r]
    all_pe = [e["prediction_error"] for r in runs for e in r]
    all_cur = [e["curiosity"] for r in runs for e in r]
    all_nm = [e["neuromod"] for r in runs for e in r]
    all_meta = [e["meta_lambda"] for r in runs for e in r]
    all_lr = [e["dynamic_lr"] for r in runs for e in r]
    all_phi = [e["phi"] for r in runs for e in r]
    all_wait = sum(1 for r in runs for e in r if e["action"] == "WAIT")
    total = sum(lifespans)
    print(f"  Lives:              {len(lifespans)}")
    print(f"  Total cycles:       {total}")
    print(f"  Avg lifespan:       {sum(lifespans)/len(lifespans):.1f} cycles")
    print(f"  Min/Max lifespan:   {min(lifespans)} / {max(lifespans)} cycles")
    print(f"  Total WAIT actions: {all_wait}/{total} cycles ({100*all_wait/total:.1f}%)")
    print(f"  Avg PE:             {sum(all_pe)/len(all_pe):.4f}")
    print(f"  Avg Curiosity:      {sum(all_cur)/len(all_cur):.4f}")
    print(f"  Avg Neuromod:       {sum(all_nm)/len(all_nm):.4f}")
    print(f"  Avg Phi:            {sum(all_phi)/len(all_phi):.4f}")
    print(f"  Avg Meta λ:         {sum(all_meta)/len(all_meta):.4f}")
    print(f"  Avg Dynamic LR:     {sum(all_lr)/len(all_lr):.6f}")
    print(f"  Action distribution: {dict(all_action_counts.most_common())}")

    # Hypothesis validation
    print(f"\n{'─' * 70}")
    print("  HYPOTHESIS: Huddling Pathology")
    print(f"{'─' * 70}")
    critical_cycles = [e for r in runs for e in r if e["battery"] < 40]
    if critical_cycles:
        wait_low = sum(1 for e in critical_cycles if e["action"] == "WAIT")
        print(f"  Cycles with battery < 40%:        {len(critical_cycles)}")
        print(f"  WAIT chosen when battery < 40%:   {wait_low} ({100*wait_low/len(critical_cycles):.1f}%)")
        print(f"  Huddling present?                  {'YES' if wait_low > len(critical_cycles)*0.2 else 'NO'}")
    else:
        print("  No cycles with battery < 40%.")

    print(f"\n  Data written to: {LOG}")

if __name__ == "__main__":
    analyze()
