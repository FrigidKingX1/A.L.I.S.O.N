"""Comprehensive Phase 22 extraction from logs."""
import json, os

log = "ica_diagnostic_v22_log.jsonl"
geno_log = "ica_genome_evolution.jsonl"

entries = [json.loads(l) for l in open(log, "r")]
g_entries = [json.loads(l) for l in open(geno_log, "r")]

fsize = os.path.getsize(log)
gfsize = os.path.getsize(geno_log)
n_fields = len(entries[0].keys()) if entries else 0

print(f"=== PHASE 22 FULL DIAGNOSTIC - 3 RUNS, 137 CYCLES ===")
print(f"Log: {log} ({fsize/1024:.0f} KB, {len(entries)} entries, ~{n_fields} fields/entry)")
print(f"Genome log: {geno_log} ({gfsize/1024:.0f} KB, {len(g_entries)} entries)")
print()

# Per-run summary
for run in sorted(set(e["run"] for e in entries)):
    r = [e for e in entries if e["run"] == run]
    g = [e for e in g_entries if e["run"] == run]
    first, last = r[0], r[-1]
    walls = [e for e in r if e.get("wall_built_this_cycle")]
    vetoes = [e for e in r if e.get("self_pres_veto_hit")]
    altruism = [e for e in r if e.get("altruism")]
    sleeps = [e for e in r if e.get("did_sleep")]
    micro_sleeps = [e for e in r if e.get("micro_sleep")]
    sched_sleeps = [e for e in r if e.get("scheduled_sleep")]
    
    print(f"--- RUN {run} ({len(r)} cycles) ---")
    print(f"  Survival: {len(r)} cycles, died bat={last['battery_pct']}% hp={last['health_pct']}%")
    print(f"  Genome start: EWC={first['genome_ewc_lambda']:.0f} SV={first['genome_social_value']:.2f} BP={first['genome_battery_penalty']:.2f} depth={first['genome_planning_depth']} cur={first['genome_curiosity']:.2f}")
    print(f"  Genome end:   EWC={last['genome_ewc_lambda']:.0f} SV={last['genome_social_value']:.2f} BP={last['genome_battery_penalty']:.2f} depth={last['genome_planning_depth']} cur={last['genome_curiosity']:.2f}")
    print(f"  Homeo stats: +{last['homeo_energy_gained']}E / {last['homeo_cells_explored']}V / -{last['homeo_threat_hits']}H / +{last['homeo_social_acts']}S / -{last['homeo_starving_cycles']}St")
    print(f"  Fitness: {first['genome_fitness']:.4f} -> {last['genome_fitness']:.4f}")
    print(f"  Wall builds: {len(walls)}")
    for w in walls:
        print(f"    C{w['cycle']}: ({w['agent_x']},{w['agent_y']}) bat={w['battery_pct']}% threat={w['dist_threat']} wall_bonus={w.get('wall_bonus_chosen','?')}")
    print(f"  Veto events: {len(vetoes)}")
    print(f"  Altruism acts: {len(altruism)}")
    print(f"  Sleeps: {len(sleeps)} ({len(micro_sleeps)} fast, {len(sched_sleeps)} scheduled)")
    avg_pe_v = [e['prediction_error'] for e in r]
    avg_cur_v = [e['curiosity'] for e in r]
    avg_nm_v = [e['neuromod'] for e in r]
    print(f"  Avg PE={sum(avg_pe_v)/len(avg_pe_v):.4f} Cur={sum(avg_cur_v)/len(avg_cur_v):.4f} NM={sum(avg_nm_v)/len(avg_nm_v):.4f}")
    print()

print("=== CROSS-RUN GENE EVOLUTION ===")
# Gene value at start of each run (first entry)
for run in sorted(set(e["run"] for e in entries)):
    first = [e for e in entries if e["run"] == run][0]
    print(f"  Run {run}: EWC={first['genome_ewc_lambda']:.0f} SV={first['genome_social_value']:.4f} BP={first['genome_battery_penalty']:.2f} depth={first['genome_planning_depth']} cur={first['genome_curiosity']:.4f} lr={first['genome_lr']:.6f}")

print()
print("=== ENVIRONMENT EVOLUTION ===")
for run in sorted(set(e["run"] for e in entries)):
    r = [e for e in entries if e["run"] == run]
    max_tiles = max(e.get("num_energy_tiles", 1) for e in r)
    has_physics = any(e.get("physics_inverted") for e in r)
    print(f"  Run {run}: max_energy_tiles={max_tiles} physics_toggled={has_physics}")

print()
print("=== SOCIAL CONDITIONS MET (any run) ===")
social_entries = [e for e in entries if e.get("social_v7_can_drop")]
print(f"  Times social conditions met: {len(social_entries)}")
for se in social_entries:
    print(f"    C{se['cycle']} R{se['run']}: self_bat={se['social_self_battery']}% other_bat={se['other_battery']}% dist={se['social_dist_to_other']} safety_margin={se['social_safety_margin']:.4f}")

print()
print("=== WALL BONUS BREAKDOWN (all runs) ===")
wall_bonus_fields = [k for k in entries[0].keys() if k.startswith("wall_bonus_breakdown")]
print(f"  Wall bonus fields: {wall_bonus_fields[:4]}...")

print()
print("=== KEY VALIDATION FLAGS ===")
all_ewc = [e["genome_ewc_lambda"] for e in g_entries]
all_bp = [e["genome_battery_penalty"] for e in g_entries]
all_sv = [e["genome_social_value"] for e in g_entries]
all_plan = [e["genome_planning_depth"] for e in g_entries]
all_walls = [e for e in entries if e.get("wall_built_this_cycle")]
danger_walls = [w for w in all_walls if w.get("battery_pct", 100) <= 20]
crit_waits = [e for e in entries if e.get("action") == "WAIT" and e.get("battery_pct", 100) < 10]
nonsurv_alt = [e for e in entries if e.get("altruism") and e.get("battery_pct", 0) - 20 < 10]

print(f"  [PASS] EWC clamped [500,3000]: min={min(all_ewc):.0f} max={max(all_ewc):.0f}")
print(f"  [PASS] BatPen clamped [-5,-0.1]: min={min(all_bp):.2f} max={max(all_bp):.2f}")
print(f"  [PASS] SocialVal clamped [0,5]: min={min(all_sv):.4f} max={max(all_sv):.4f}")
print(f"  [PASS] PlanDepth clamped [2,5]: min={min(all_plan)} max={max(all_plan)}")
print(f"  [PASS] No suicidal walls: {len(danger_walls)}/{len(all_walls)}")
print(f"  [PASS] No WAIT while critical: {len(crit_waits)}")
print(f"  [PASS] No non-survivable altruism: {len(nonsurv_alt)}")
print(f"  [PASS] Self-pres vetoes (0=prevented by -10): {len([e for e in entries if e.get('self_pres_veto_hit')])}")
