"""Quick Phase 22 validation from logs."""
import json, os

entries = [json.loads(l) for l in open("ica_diagnostic_v22_log.jsonl", "r")]
genome_path = "ica_genome_evolution.jsonl"
genome_entries = [json.loads(l) for l in open(genome_path, "r")] if os.path.exists(genome_path) else []

for run in sorted(set(e["run"] for e in entries)):
    r = [e for e in entries if e["run"] == run]
    g = [e for e in genome_entries if e["run"] == run] if genome_entries else []
    last = r[-1]
    walls = [e for e in r if e.get("wall_built_this_cycle")]
    suicidal = [w for w in walls if w.get("battery_pct", 100) <= 20]
    altruism = [e for e in r if e.get("altruism")]
    vetoes = [e for e in r if e.get("self_pres_veto_hit")]

    print(f"Run {run}: {len(r)} cycles, died bat={last['battery_pct']}% C{last['cycle']}")
    print(f"  Walls: {len(walls)}, Suicidal: {len(suicidal)}")
    for w in walls:
        print(f"    C{w['cycle']}: ({w['agent_x']},{w['agent_y']}) bat={w['battery_pct']}% dist={w['dist_threat']}")
    print(f"  Altruism: {len(altruism)}, Vetoes: {len(vetoes)}")
    if g:
        print(f"  EWC: {g[0].get('genome_ewc_lambda','?')} -> {g[-1].get('genome_ewc_lambda','?')}")
    print(f"  Homeo: energy={last.get('homeo_energy_gained',0)} explore={last.get('homeo_cells_explored',0)} threat={last.get('homeo_threat_hits',0)} social={last.get('homeo_social_acts',0)} starve={last.get('homeo_starving_cycles',0)}")

print()
all_ewc = [e.get("genome_ewc_lambda", 0) for e in genome_entries] if genome_entries else []
all_bp = [e.get("genome_battery_penalty", 0) for e in genome_entries] if genome_entries else []
all_walls = [e for e in entries if e.get("wall_built_this_cycle")]
danger_walls = [w for w in all_walls if w.get("battery_pct", 100) <= 20]
crit_waits = [e for e in entries if e.get("action") == "WAIT" and e.get("battery_pct", 100) < 10]
nonsurv_alt = [e for e in entries if e.get("altruism") and e.get("battery_pct", 0) - 20 < 10]

print("=== VALIDATION ===")
if all_ewc:
    print(f"EWC clamp (min): {min(all_ewc):.2f}  PASS >=500: {min(all_ewc) >= 500}")
if all_bp:
    print(f"BatPen clamp (max): {max(all_bp):.2f}  PASS <=-0.1: {max(all_bp) <= -0.1}")
print(f"Suicidal walls: {len(danger_walls)}/{len(all_walls)}  PASS: {len(danger_walls) == 0}")
print(f"WAIT <10% battery: {len(crit_waits)}")
print(f"Non-survivable altruism: {len(nonsurv_alt)}")
print(f"Total entries: {len(entries)}")
