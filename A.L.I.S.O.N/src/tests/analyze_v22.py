"""Analyze Phase 22 diagnostic results."""
import json

log = "ica_diagnostic_v22_log.jsonl"
genome_log = "ica_genome_evolution.jsonl"

entries = [json.loads(l) for l in open(log, "r")]
genome_entries = [json.loads(l) for l in open(genome_log, "r")]

runs = sorted(set(e["run"] for e in entries))
print(f"Total entries: {len(entries)} across runs: {runs}")

for run in runs:
    r_entries = [e for e in entries if e["run"] == run]
    r_genome = [e for e in genome_entries if e["run"] == run]
    last_e = r_entries[-1]
    first_g = r_genome[0] if r_genome else {}
    last_g = r_genome[-1] if r_genome else {}

    wall_events = [e for e in r_entries if e.get("wall_built_this_cycle")]
    veto_events = [e for e in r_entries if e.get("self_pres_veto_hit")]
    altruism_acts = [e for e in r_entries if e.get("altruism")]
    wall_below_20 = [e for e in wall_events if e.get("battery_pct", 100) <= 20]

    print(f'\n=== RUN {run} ({len(r_entries)} cycles) ===')
    print(f"  Death: battery={last_e['battery_pct']}% at C{last_e['cycle']}")
    print(f"  Wall builds: {len(wall_events)}, Below 20% bat: {len(wall_below_20)}")
    for w in wall_events:
        print(f"    C{w['cycle']}: ({w['agent_x']},{w['agent_y']}) bat={w['battery_pct']}% threat_dist={w['dist_threat']} wall_bonus={w.get('wall_bonus_chosen',0)}")
    print(f"  Self-pres vetoes: {len(veto_events)}")
    for v in veto_events:
        print(f"    C{v['cycle']}: {v.get('self_pres_veto_reason','?')}")
    print(f"  Altruism acts: {len(altruism_acts)}")
    for a in altruism_acts:
        print(f"    C{a['cycle']}: bat={a['battery_pct']}% safety_margin={a.get('drop_safety_margin',0):.4f}")
    print(f"  Gene start: ewc={first_g.get('genome_ewc_lambda','?')} social={first_g.get('genome_social_value','?')} bat_pen={first_g.get('genome_battery_penalty','?')} depth={first_g.get('genome_planning_depth','?')}")
    print(f"  Gene end:   ewc={last_g.get('genome_ewc_lambda','?')} social={last_g.get('genome_social_value','?')} bat_pen={last_g.get('genome_battery_penalty','?')} depth={last_g.get('genome_planning_depth','?')}")
    print(f"  Homeo stats: energy={last_e.get('homeo_energy_gained',0)} explored={last_e.get('homeo_cells_explored',0)} hits={last_e.get('homeo_threat_hits',0)} social_acts={last_e.get('homeo_social_acts',0)} starve={last_e.get('homeo_starving_cycles',0)}")

# CROSS-RUN VALIDATION
print("\n=== CROSS-RUN VALIDATION ===")

# 1. EWC clamp
all_ewc = [e["genome_ewc_lambda"] for e in genome_entries]
print(f"EWC range: {min(all_ewc):.2f} to {max(all_ewc):.2f}")
print(f"  PASS >= 500: {all(v >= 500 for v in all_ewc)}")
print(f"  PASS <= 3000: {all(v <= 3000 for v in all_ewc)}")

# 2. Battery penalty clamp
all_bp = [e["genome_battery_penalty"] for e in genome_entries]
print(f"BatPen range: {min(all_bp):.2f} to {max(all_bp):.2f}")
print(f"  PASS >= -5.0: {all(v >= -5.0 for v in all_bp)}")
print(f"  PASS <= -0.1: {all(v <= -0.1 for v in all_bp)}")

# 3. No suicidal walls
all_walls = [e for e in entries if e.get("wall_built_this_cycle")]
dangerous_walls = [w for w in all_walls if w.get("battery_pct", 100) <= 20]
print(f"Total walls: {len(all_walls)}, Suicidal walls (bat<=20%): {len(dangerous_walls)}")
print(f"  PASS (no suicidal walls): {len(dangerous_walls) == 0}")

# 4. No suicidal WAIT
critical_waits = [e for e in entries if e.get("action") == "WAIT" and e.get("battery_pct", 100) < 10]
print(f"WAIT while <10%: {len(critical_waits)}")

# 5. Altruism safety
all_altruism = [e for e in entries if e.get("altruism")]
nonsurvivable = [a for a in all_altruism if a.get("battery_pct", 0) - 20 < 10]
print(f"Altruism acts: {len(all_altruism)}, Non-survivable: {len(nonsurvivable)}")
for a in nonsurvivable:
    bat_before = a.get("battery_pct", 0)
    print(f"  C{a['cycle']}: bat before={bat_before}% (drop to {bat_before-20}%) other_starving={a.get('other_starving','?')}")
print(f"  PASS (no non-survivable altruism): {len(nonsurvivable) == 0}")

# 6. HOMEOSTATIC FITNESS TRACKING
print("\n--- HOMEOSTATIC FITNESS ---")
for run in runs:
    r_entries = [e for e in entries if e["run"] == run]
    last = r_entries[-1]
    print(f"  Run {run}: fitness={last.get('homeo_fitness_now',0):.4f} stats={last.get('homeo_energy_gained',0)}E/{last.get('homeo_cells_explored',0)}V/{last.get('homeo_threat_hits',0)}H/{last.get('homeo_social_acts',0)}S/{last.get('homeo_starving_cycles',0)}St")
