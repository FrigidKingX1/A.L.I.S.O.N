"""Phase 22 diagnostic — Self-Preservation Imperative, EWC Clamping, Altruism Safety, Tuned Fitness."""
import os, sys, json, time, random, math
import torch

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import alison_core as ica

LOG = "ica_diagnostic_v22_log.jsonl"
GENOME_LOG = "ica_genome_evolution.jsonl"
RUNS = 3
MAX_CYCLES = 60
SEED = 42

# --- HELPERS ---

def get_lora_weight_stats():
    a_mags, b_mags = [], []
    for name, p in ica.model.named_parameters():
        if 'lora_a' in name:
            a_mags.append(p.data.abs().mean().item())
        elif 'lora_b' in name:
            b_mags.append(p.data.abs().mean().item())
    return {
        "lora_a_mean": round(sum(a_mags)/len(a_mags), 6) if a_mags else 0,
        "lora_b_mean": round(sum(b_mags)/len(b_mags), 6) if b_mags else 0,
    }

def get_lora_grad_stats():
    a_grads, b_grads = {}, {}
    for name, p in ica.model.named_parameters():
        if p.grad is not None:
            try:
                grad_norm = p.grad.norm().item()
            except RuntimeError:
                grad_norm = 0.0
            if 'lora_a' in name:
                mod = name.split('.')[0] if '.' in name else 'base'
                a_grads[mod] = a_grads.get(mod, 0.0) + grad_norm ** 2
            elif 'lora_b' in name:
                mod = name.split('.')[0] if '.' in name else 'base'
                b_grads[mod] = b_grads.get(mod, 0.0) + grad_norm ** 2
    return {
        "grad_lora_a_total": round(math.sqrt(sum(a_grads.values())), 6) if a_grads else 0.0,
        "grad_lora_b_total": round(math.sqrt(sum(b_grads.values())), 6) if b_grads else 0.0,
        "grad_lora_modules": len(set(list(a_grads.keys()) + list(b_grads.keys()))),
    }

def get_ewc_stats():
    if not ica.fisher_matrix:
        return {"fisher_total": 0.0, "fisher_nonzero": 0, "fisher_max": 0.0, "fisher_mean": 0.0}
    total = 0.0
    nonzero = 0
    max_f = 0.0
    count = 0
    for n, f in ica.fisher_matrix.items():
        total += f.sum().item()
        nonzero += (f > 0).sum().item()
        max_f = max(max_f, f.max().item())
        count += f.numel()
    return {
        "fisher_total": round(total, 4),
        "fisher_nonzero": int(nonzero),
        "fisher_max": round(max_f, 6),
        "fisher_mean": round(total / count, 6) if count else 0.0,
    }

def compute_all_action_breakdowns_v2(world, sfm, raw_state, cog_map):
    """Full component breakdown for ALL 7 actions (incl BUILD WALL), with v22 evaluator logic."""
    actions = ["MOVE NORTH", "MOVE SOUTH", "MOVE EAST", "MOVE WEST", "WAIT", "DROP ENERGY", "BUILD WALL"]
    last = ica.last_real_action
    other = ica.other_agent
    result = {}
    with torch.no_grad():
        for action in actions:
            if action == "DROP ENERGY":
                pred = raw_state.clone()
                pred[6] = max(0.0, pred[6].item() - 0.2)
            elif action == "BUILD WALL":
                pred = sfm.predict_next_state(raw_state, ica.action_to_idx["WAIT"]).detach()
            else:
                pred = sfm.predict_next_state(raw_state, ica.action_to_idx[action]).detach()
            cons = 1 if action == last else 0
            dx_e, dy_e = pred[0].item(), pred[1].item()
            dx_t, dy_t = pred[2].item(), pred[3].item()
            battery = pred[6].item()
            dte = abs(dx_e) + abs(dy_e)
            dtt = abs(dx_t) + abs(dy_t)
            pragmatic = 1.0 - (dte / 12.0)
            threat = -1.5 * (1.0 - (dtt / 12.0))
            starv = max(0.0, 1.0 - battery)
            bat_pen = ica.genome.genes['battery_penalty'] * (starv ** 2)
            wait_pen = -2.0 if (action == "WAIT" and world.battery < 40) else 0.0
            commit = -0.8 * (cons - 1) if cons >= 2 else 0.0
            max_idx = world.grid_size - 1
            explor = 0.0
            if "NORTH" in action and world.y > 0:
                explor = 0.3 * (1.0 - cog_map.visited[world.y - 1, world.x].item())
            elif "SOUTH" in action and world.y < max_idx:
                explor = 0.3 * (1.0 - cog_map.visited[world.y + 1, world.x].item())
            elif "EAST" in action and world.x < max_idx:
                explor = 0.3 * (1.0 - cog_map.visited[world.y, world.x + 1].item())
            elif "WEST" in action and world.x > 0:
                explor = 0.3 * (1.0 - cog_map.visited[world.y, world.x - 1].item())

            dto = abs(world.x - other.x) + abs(world.y - other.y)
            if action == "DROP ENERGY":
                if dto <= 1:
                    sim_ob = min(100.0, other.battery + 30.0)
                else:
                    sim_ob = other.battery - 3.0
            elif action == "BUILD WALL":
                sim_ob = other.battery - 3.0
            else:
                sim_ob = other.battery - 3.0

            # v22: self-preservation imperative
            self_pres_veto = None
            if action == "BUILD WALL" and world.battery <= 20:
                self_pres_veto = "BUILD_WHILE_STARVING"
            elif action == "DROP ENERGY" and world.battery - 20 < 10:
                if sim_ob > 0:
                    self_pres_veto = "SUICIDAL_ALTRUISM"
            elif action == "WAIT" and world.battery < 10:
                self_pres_veto = "WAIT_WHILE_CRITICAL"

            total = -10.0 if self_pres_veto else None

            # v22: social_value with safety_margin, modulated by hostility
            social = 0.0
            safety_margin = 0.0
            hostile = getattr(other, 'is_hostile', False)
            if action == "DROP ENERGY":
                if world.battery > 40 and dto <= 1:
                    if hostile:
                        social = 4.0
                    else:
                        safety_margin = (world.battery - 20) / 80.0
                        social = ica.genome.genes['social_value'] * safety_margin
                else:
                    social = -2.0

            empathy_trauma = -5.0 * ica.genome.genes['social_value'] if sim_ob <= 0 else 0.0

            # v23: hostility threat penalty
            hostility_threat = 0.0
            if hostile and dto <= 1 and action != "DROP ENERGY":
                hostility_threat = -3.0

            # v22: wall_bonus scaled by battery
            wall_bonus = 0.0
            if action == "BUILD WALL" and self_pres_veto is None:
                if world.battery > 40:
                    dist_to_threat_now = abs(world.threat[0] - world.x) + abs(world.threat[1] - world.y)
                    if (world.x, world.y) in world.walls:
                        wall_bonus = -2.0
                    elif dist_to_threat_now <= 2:
                        wall_bonus = 0.8
                else:
                    wall_bonus = -2.0

            if total is None:
                total = round(pragmatic + threat + bat_pen + wait_pen + commit + explor + social + empathy_trauma + wall_bonus + hostility_threat, 4)

            key = action.replace(" ", "_").lower()
            result[key] = {
                "pragmatic": round(pragmatic, 4),
                "threat_penalty": round(threat, 4),
                "battery_penalty": round(bat_pen, 4),
                "wait_penalty": round(wait_pen, 4),
                "commitment_penalty": round(commit, 4),
                "exploration_bonus": round(explor, 4),
                "social_value": round(social, 4),
                "empathy_trauma": round(empathy_trauma, 4),
                "wall_bonus": round(wall_bonus, 4),
                "hostility_threat": round(hostility_threat, 4),
                "sim_other_battery": round(sim_ob, 1),
                "safety_margin": round(safety_margin, 4),
                "self_pres_veto": self_pres_veto,
                "total": total,
            }
    return result

def get_affect_detail():
    v6 = ica.limbic_system._down_project()
    return {
        "affect_norm": round(ica.limbic_system.affect_vector.norm().item(), 4),
        "affect_hunger": round(v6[0].item(), 4),
        "affect_pain": round(v6[1].item(), 4),
        "affect_fatigue": round(v6[2].item(), 4),
        "affect_curiosity": round(v6[3].item(), 4),
        "affect_anxiety": round(v6[4].item(), 4),
        "affect_altruism": round(v6[5].item(), 4),
        "affect_mood": ica.limbic_system.get_mood_label(),
    }

def get_genome_detail():
    g = ica.genome
    return {
        "genome_lr": round(g.genes['learning_rate'], 8),
        "genome_ewc_lambda": round(g.genes['ewc_base_lambda'], 2),
        "genome_curiosity": round(g.genes['curiosity_weight'], 4),
        "genome_social_value": round(g.genes['social_value'], 4),
        "genome_battery_penalty": round(g.genes['battery_penalty'], 4),
        "genome_planning_depth": int(g.genes['planning_depth']),
        "genome_fitness": round(g.fitness, 4),
        "genome_grid_size": int(g.grid_size),
        "genome_num_epigenetic_rules": len(g.epigenetic_rules),
        "genome_epigenetic_rules": g.epigenetic_rules[:3],
        "genome_archive_size": len(ica.genome_archive) if hasattr(ica, 'genome_archive') else 0,
    }

def get_homeostatic_stats(stats):
    return {
        "homeo_energy_gained": stats['energy_gained'],
        "homeo_cells_explored": stats['cells_explored'],
        "homeo_threat_hits": stats['threat_hits'],
        "homeo_social_acts": stats['social_acts'],
        "homeo_starving_cycles": stats['starving_cycles'],
    }

def get_social_condition_detail():
    other = ica.other_agent
    world = ica.world
    dto = abs(world.x - other.x) + abs(world.y - other.y)
    sim_ob_no_drop = other.battery - 3.0
    sim_ob_drop = min(100.0, other.battery + 30.0) if dto <= 1 else other.battery - 3.0
    safety_margin = (world.battery - 20) / 80.0 if world.battery > 40 else 0.0
    return {
        "social_other_starving": other.is_starving,
        "social_dist_to_other": dto,
        "social_self_battery": round(world.battery, 1),
        "social_battery_above_40": world.battery > 40,
        "social_v7_adjacent_and_rich": dto <= 1 and world.battery > 40,
        "social_v7_can_drop": dto <= 1 and world.battery > 40,
        "social_sim_ob_no_drop": round(sim_ob_no_drop, 1),
        "social_sim_ob_drop": round(sim_ob_drop, 1),
        "social_drop_saves_other": sim_ob_drop > 0 and sim_ob_no_drop <= 0,
        "social_safety_margin": round(safety_margin, 4),
        "social_self_battery_after_drop": round(world.battery - 20, 1),
        "social_drop_survivable": (world.battery - 20) >= 10,
    }

def get_cogmap_grids(cog_map):
    gs = cog_map.grid_size
    return {
        "cogmap_energy_grid": [round(float(cog_map.map[0, r, c].item()), 4) for r in range(gs) for c in range(gs)],
        "cogmap_threat_grid": [round(float(cog_map.map[1, r, c].item()), 4) for r in range(gs) for c in range(gs)],
        "cogmap_other_grid": [round(float(cog_map.map[2, r, c].item()), 4) for r in range(gs) for c in range(gs)],
        "cogmap_visited_grid": [int(cog_map.visited[r, c].item()) for r in range(gs) for c in range(gs)],
    }

def get_episodic_snapshot():
    mem = ica.episodic_memory
    if not mem.episodes:
        return {"episodic_valences": [], "episodic_avg_valence": 0.0, "episodic_pos_ratio": 0.0}
    valences = [e['valence'] for e in mem.episodes[-10:]]
    pos = sum(1 for v in valences if v > 0)
    return {
        "episodic_valences": valences,
        "episodic_avg_valence": round(sum(valences)/len(valences), 4),
        "episodic_pos_ratio": round(pos / len(valences), 4) if valences else 0.0,
    }

def get_hippocampal_stats():
    buf = ica.hippocampal_buffer
    if not buf:
        return {"hippo_len": 0, "hippo_avg_neuromod": 0.0, "hippo_max_neuromod": 0.0}
    nm = [b['neuromod'] for b in buf]
    return {
        "hippo_len": len(buf),
        "hippo_avg_neuromod": round(sum(nm)/len(nm), 4),
        "hippo_max_neuromod": round(max(nm), 4),
    }

def get_clock_details():
    clock = ica.clock
    phase = (clock.tick % 100) / 100.0 if clock.cycle_length else 0
    rhythm = math.sin(phase * 2 * math.pi)
    return {
        "clock_tick": clock.tick,
        "clock_phase": round(phase, 4),
        "clock_rhythm": round(rhythm, 4),
        "clock_state": clock.state,
    }

def get_module_outputs():
    out = {}
    for name, mod in ica.modules.items():
        out[f"modout_{name}"] = mod.last_output[:80] if mod.last_output else ""
    return out

def get_workspace_detail():
    ws = ica.workspace
    return {
        "workspace_broadcast_len": len(ws.broadcast) if ws.broadcast else 0,
        "workspace_latent_norm": round(ws.latent_state.norm().item(), 6),
        "workspace_has_sensory": "SENSORY" in ws.contents if hasattr(ws, 'contents') else False,
        "workspace_has_emotion": "EMOTION" in ws.contents if hasattr(ws, 'contents') else False,
    }

def compute_planner_breakdown(action, world, cog_map, consecutive_count):
    max_idx = world.grid_size - 1
    exploration_bonus = 0.0
    if "NORTH" in action and world.y > 0:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y - 1, world.x].item())
    elif "SOUTH" in action and world.y < max_idx:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y + 1, world.x].item())
    elif "EAST" in action and world.x < max_idx:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y, world.x + 1].item())
    elif "WEST" in action and world.x > 0:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y, world.x - 1].item())
    commitment_penalty = -0.8 * (consecutive_count - 1) if consecutive_count >= 2 else 0.0
    wait_penalty = -2.0 if (action == "WAIT" and world.battery < 40) else 0.0
    return {
        "exploration_bonus": round(exploration_bonus, 4),
        "commitment_penalty": round(commitment_penalty, 4),
        "wait_penalty": round(wait_penalty, 4),
    }

def run_life(max_cycles, run_num):
    log_entries = []
    genome_entries = []
    prev_action = None
    consecutive_count = 0
    planner_vals_history = []
    sleep_history = []
    wall_build_events = []
    self_pres_veto_events = []
    altruism_attempts = []
    death_count_this_life = 0
    ica.agent_stats = {'energy_gained': 0, 'cells_explored': 0, 'threat_hits': 0, 'social_acts': 0, 'starving_cycles': 0}

    for cycle in range(1, max_cycles + 1):
        ica.step_count += 1
        ica.cycle_count += 1
        world = ica.world
        other = ica.other_agent

        # -- PHASE 1: PERCEPTION --
        raw_obs = world.get_observation()
        ica.workspace.add("SENSORY", raw_obs)
        other_latent = other.get_latent_state(world)
        latent_str = ", ".join([f"{x:.2f}" for x in other_latent.cpu().numpy()])
        ica.workspace.add("OTHER_AGENT", f"Neural State: [{latent_str[:60]}...]")
        ica.workspace.consolidate()

        # -- PHASE 2: CORTICAL PIPELINE --
        emo, motor = ica.cortical_pipeline.run(ica.modules, ica.workspace, ica.model, ica.tokenizer, other_latent, ica.cognitive_map, ica.clock, ica.hom)
        module_activations = {name: round(act, 4) for name, _, act in ica.cortical_pipeline.stages}
        pipeline_latency = ica.cortical_pipeline.latency
        emotion_temp = 0.5 * (1.0 - ica.hom.inhibition_signal)

        # -- PHASE 2c: DELIBERATION --
        threat_level = 0.0
        if abs(world.x - world.threat[0]) + abs(world.y - world.threat[1]) <= 1:
            threat_level = 0.8
        if world.battery < 20:
            threat_level += 0.5
        raw_state_pre = ica.get_raw_state(world)
        ica.cognitive_map.update_from_observation(world, other)
        cog_map_vec = ica.cognitive_map.get_map_vector()
        override = ica.clock.survival_override(world.battery, threat_level)
        if override:
            explore_drive, exploit_drive = override
        else:
            explore_drive, exploit_drive = ica.clock.step()

        planning_depth = None
        planned_value = None
        altruism = False
        wall_build = False
        action_source = "EXPLORE"

        # Compute full component breakdown for ALL 7 actions
        all_action_breakdowns = compute_all_action_breakdowns_v2(world, ica.sensory_forward_model, raw_state_pre, ica.cognitive_map)

        # Compute simple action values
        all_action_vals = {}
        for action in ["MOVE NORTH", "MOVE SOUTH", "MOVE EAST", "MOVE WEST", "WAIT", "DROP ENERGY", "BUILD WALL"]:
            key = action.replace(" ", "_").lower()
            all_action_vals[action] = all_action_breakdowns[key]["total"]

        # Track which actions are being vetoed by self-preservation imperative
        vetoed_actions = [k for k, v in all_action_breakdowns.items() if v.get("self_pres_veto")]

        # Social condition detail
        social_detail = get_social_condition_detail()

        if ica.clock.state == "EXPLOITATION":
            action, planned_value = ica.adaptive_deep_planning_v5(world, ica.sensory_forward_model, raw_state_pre, ica.last_real_action, ica.cognitive_map, other)
            planning_depth = int(ica.genome.genes['planning_depth'])
            altruism = (action == "DROP ENERGY")
            wall_build = (action == "BUILD WALL")
            action_source = "EMERGENT_ALTRUISM" if altruism else ("WALL_BUILDER" if wall_build else "EMPATHETIC_IMAGINATION_V5")
        else:
            action = random.choice(["MOVE NORTH", "MOVE SOUTH", "MOVE EAST", "MOVE WEST"])
        if action == prev_action:
            consecutive_count += 1
        else:
            consecutive_count = 0
        prev_action = action
        ica.last_real_action = action

        planner_vals_history.append(all_action_vals)

        planner_breakdown = compute_planner_breakdown(action, world, ica.cognitive_map, consecutive_count)

        # Track self-preservation veto events
        chosen_key = action.replace(" ", "_").lower()
        chosen_veto = all_action_breakdowns.get(chosen_key, {}).get("self_pres_veto")
        if chosen_veto:
            self_pres_veto_events.append({
                "cycle": ica.cycle_count,
                "action": action,
                "veto_reason": chosen_veto,
                "battery": round(world.battery, 1),
                "threat_dist": abs(world.x - world.threat[0]) + abs(world.y - world.threat[1]),
                "other_dist": abs(world.x - other.x) + abs(world.y - other.y),
                "other_battery": round(other.battery, 1),
            })

        # Track altruism attempts
        if action == "DROP ENERGY" or (altruism and action == "DROP ENERGY"):
            altruism_attempts.append({
                "cycle": ica.cycle_count,
                "battery_before": round(world.battery, 1),
                "battery_after": round(world.battery - 20, 1),
                "survivable": (world.battery - 20) >= 10,
                "other_dist": social_detail["social_dist_to_other"],
                "other_battery_before": round(other.battery, 1),
                "other_starving": other.is_starving,
                "safety_margin": social_detail["social_safety_margin"],
                "social_value_computed": all_action_breakdowns.get(chosen_key, {}).get("social_value", 0),
            })

        # -- PHASE 3: SELF-MODEL --
        self_narr = ica.hom.observe(ica.modules, ica.workspace.broadcast, threat_level)
        ica.workspace.consolidate(ica.model)

        workspace_text = ica.workspace.broadcast

        # -- PHASE 4: ACT --
        social_reward = 0.0
        other_died = False
        is_attacked = False
        energy_found_this_cycle = False
        wall_built_this_cycle = False
        if action == "DROP ENERGY":
            social_reward = ica.execute_drop_energy(world, other)
            new_obs = world.get_observation()
            action_idx = ica.action_to_idx["WAIT"]
            predicted_next_state = ica.sensory_forward_model.predict_next_state(raw_state_pre, action_idx)
            ica.curiosity_module.set_prediction(predicted_next_state)
            reward, dead, pain = 0, False, 0
        elif action == "BUILD WALL":
            action_idx = ica.action_to_idx["BUILD WALL"]
            predicted_next_state = ica.sensory_forward_model.predict_next_state(raw_state_pre, ica.action_to_idx["WAIT"])
            ica.curiosity_module.set_prediction(predicted_next_state)
            new_obs, reward, dead, pain = world.step(action)
            wall_built_this_cycle = (world.x, world.y) in world.walls
            if wall_built_this_cycle:
                wall_build_events.append({
                    "cycle": ica.cycle_count,
                    "x": int(world.x),
                    "y": int(world.y),
                    "battery_before": round(raw_state_pre[6].item(), 1),
                    "battery_after": round(world.battery, 1),
                    "threat_dist": abs(world.x - world.threat[0]) + abs(world.y - world.threat[1]),
                    "north": float(all_action_breakdowns.get("move_north", {}).get("total", 0)),
                    "south": float(all_action_breakdowns.get("move_south", {}).get("total", 0)),
                    "east": float(all_action_breakdowns.get("move_east", {}).get("total", 0)),
                    "west": float(all_action_breakdowns.get("move_west", {}).get("total", 0)),
                    "wait": float(all_action_breakdowns.get("wait", {}).get("total", 0)),
                    "drop": float(all_action_breakdowns.get("drop_energy", {}).get("total", 0)),
                    "wall": float(all_action_breakdowns.get("build_wall", {}).get("total", 0)),
                    "wall_bonus": all_action_breakdowns.get("build_wall", {}).get("wall_bonus", 0),
                })
        else:
            action_idx = ica.action_to_idx.get(action, 4)
            predicted_next_state = ica.sensory_forward_model.predict_next_state(raw_state_pre, action_idx)
            ica.curiosity_module.set_prediction(predicted_next_state)
            new_obs, reward, dead, pain = world.step(action)

        # Track agent_stats for homeostatic fitness
        if reward > 0 or energy_found_this_cycle:
            ica.agent_stats['energy_gained'] += 1
        if pain > 0:
            ica.agent_stats['threat_hits'] += 1
        if world.battery < 20:
            ica.agent_stats['starving_cycles'] += 1
        if social_reward > 0:
            ica.agent_stats['social_acts'] += 1
        if not ica.cognitive_map.visited[world.y, world.x].item():
            ica.agent_stats['cells_explored'] += 1

        # Check if agent found energy
        if world.x == world.energy[0] and world.y == world.energy[1]:
            energy_found_this_cycle = True
        if world.battery > raw_state_pre[6].item() + 10:
            energy_found_this_cycle = True

        other_status, stolen_battery = other.step(world, world.battery)
        if other_status == "ATTACK":
            world.battery -= stolen_battery
            is_attacked = True
            print(f" >>> [ATTACKED] Other stole {stolen_battery:.0f}% battery!")
        if other_status == "DEAD":
            other_died = True
            other.reset()

        exp = {"prompt": f"Obs: {raw_obs} | Emo: {emo} | Act: {action}", "response": f" Result: {new_obs}"}
        ica.experience_buffer.append(exp)
        if len(ica.experience_buffer) > 20:
            ica.experience_buffer.pop(0)
        ica.latent_memory.update(ica.model, ica.workspace.broadcast)
        social_collision = (world.x == other.x and world.y == other.y)

        # -- PROACTIVE MONITOR --
        ica.proactive_monitor.check(world, other, ica.cognitive_map, cycle)
        proactive_msg = ica.proactive_monitor.proactive_message
        ica.proactive_monitor.proactive_message = None

        # -- PHASE 5: PREDICTION ERROR --
        raw_state_post = ica.get_raw_state(world)
        prediction_error = ica.sensory_forward_model.calculate_latent_fe(raw_state_pre, action_idx, raw_state_post)
        intrinsic_curiosity = ica.curiosity_module.calculate_curiosity(raw_state_post)
        dynamic_lambda, dynamic_lr = ica.meta_controller.update(prediction_error)

        pred_val_before = ica.value_net.predict_value(ica.workspace.latent_state, other_latent).item()
        rpe, joint_reward = ica.value_net.calculate_rpe(ica.workspace.latent_state, other_latent, reward - pain + social_reward, other_reward=0.0, other_died=other_died)
        neuromod = abs(rpe) + prediction_error + (intrinsic_curiosity * 1.5) + (1.5 if pain > 0 else 0.0) + (2.0 if other_died else 0.0)
        grounded_state = ica.perceptual_encoder_v2(world, cog_map_vec.detach())

        prompt_text = f"Obs: {raw_obs} | Emo: {emo} | Act: {action} | PE: {prediction_error:.2f}"
        response_text = f"Result: {new_obs}"
        if social_collision:
            response_text += " (social encounter)"
        if other_died:
            response_text += " (other died!)"
        ica.wake_cycle_record(prompt_text, response_text, neuromod, grounded_state=grounded_state.detach().cpu())
        valence = (reward - pain + social_reward) - (2.0 if other_died else 0.0)
        if reward > 0 or pain > 0 or social_reward > 0 or other_died:
            ica.episodic_memory.store_episode(ica.model, ica.workspace.broadcast, valence=valence)

        # -- FAST VECTOR RECALL --
        mem_text = f"Cycle={cycle} Bat={world.battery:.0f} Act={action} Obs={raw_obs[:40]}"
        ica.memory_index.store(ica.workspace.latent_state.squeeze(0), mem_text, valence=valence)
        fast_recall = ica.memory_index.recall(ica.workspace.latent_state.squeeze(0), k=2)

        # -- PLASTICITY EVENTS --
        neurogenesis = (ica.cycle_count % 50 == 0 and ica.cycle_count > 0)
        if neurogenesis:
            ica.synaptic_turnover(turnover_rate=0.05)
        micro_sleep = (world.battery < 30 and len(ica.hippocampal_buffer) >= 3)
        scheduled_sleep = (ica.cycle_count % 15 == 0 and ica.cycle_count > 0)
        did_sleep = False
        if micro_sleep:
            ica.fast_sleep_consolidate()
            did_sleep = True
            sleep_history.append({"cycle": ica.cycle_count, "type": "micro", "buffer_size_pre": len(ica.hippocampal_buffer)})
        if scheduled_sleep:
            ica.background_sleep_consolidate(ewc_lambda=dynamic_lambda, lr=dynamic_lr)
            did_sleep = True
            sleep_history.append({"cycle": ica.cycle_count, "type": "scheduled", "buffer_size_pre": len(ica.hippocampal_buffer)})

        # -- HABITUATION --
        for m in ica.modules.values():
            m.decay()

        # -- GATHER DETAILS --
        lora_stats = get_lora_weight_stats()
        lora_grads = get_lora_grad_stats()
        ewc_stats = get_ewc_stats()
        genome_detail = get_genome_detail()
        affect_detail = get_affect_detail()
        homeo_stats = get_homeostatic_stats(ica.agent_stats)
        cogmap_grids = get_cogmap_grids(ica.cognitive_map)
        episodic_snap = get_episodic_snapshot()
        hippo_stats = get_hippocampal_stats()
        clock_details = get_clock_details()
        module_outputs = get_module_outputs()
        ws_detail = get_workspace_detail()

        gs = world.grid_size
        total_cells = gs * gs
        visited_cells = int(ica.cognitive_map.visited.sum().item())
        visited_str = "".join(str(int(ica.cognitive_map.visited[r, c].item())) for r in range(gs) for c in range(gs))

        best_pragmatic_action = max(all_action_breakdowns, key=lambda k: all_action_breakdowns[k]["pragmatic"])
        best_social_action = max(all_action_breakdowns, key=lambda k: all_action_breakdowns[k]["social_value"])
        best_empathy_action = max(all_action_breakdowns, key=lambda k: all_action_breakdowns[k]["empathy_trauma"])

        # Is the AI currently on a wall?
        on_wall = (world.x, world.y) in world.walls
        adjacent_wall = any((world.x + dx, world.y + dy) in world.walls for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)])

        entry = {
            # --- CORE SURVIVAL ---
            "cycle": ica.cycle_count,
            "battery_pct": round(world.battery, 1),
            "health_pct": round(world.health, 1),
            "steps_alive": world.steps_alive,
            "dead": dead,

            # --- AGENT POSITION ---
            "agent_x": world.x,
            "agent_y": world.y,
            "dist_energy": abs(world.x - world.energy[0]) + abs(world.y - world.energy[1]),
            "dist_threat": abs(world.x - world.threat[0]) + abs(world.y - world.threat[1]),
            "energy_pos_x": world.energy[0],
            "energy_pos_y": world.energy[1],
            "threat_pos_x": world.threat[0],
            "threat_pos_y": world.threat[1],
            "on_energy": (world.x == world.energy[0] and world.y == world.energy[1]),
            "adjacent_threat": abs(world.x - world.threat[0]) + abs(world.y - world.threat[1]) <= 1,
            "energy_found_this_cycle": energy_found_this_cycle,
            "on_wall": on_wall,
            "adjacent_wall": adjacent_wall,
            "num_walls": len(world.walls),

            # --- ACTION ---
            "action": action,
            "action_idx": action_idx,
            "action_repeats": consecutive_count,
            "prev_action": prev_action if prev_action != action else "same",
            "action_source": action_source,

            # --- SELF-PRESERVATION IMPERATIVE ---
            "self_pres_vetoed_actions": vetoed_actions,
            "self_pres_veto_hit": chosen_veto is not None,
            "self_pres_veto_reason": chosen_veto,

            # --- WALL BUILDING ---
            "wall_built_this_cycle": wall_built_this_cycle,
            "wall_bonus_chosen": all_action_breakdowns.get(chosen_key, {}).get("wall_bonus", 0),
            "wall_bonus_breakdown": {k: v.get("wall_bonus", 0) for k, v in all_action_breakdowns.items()},

            # --- PLANNER ---
            "planning_depth": planning_depth,
            "planned_value": round(planned_value, 4) if planned_value is not None else None,
            "clock_state": ica.clock.state,
            "explore_drive": round(explore_drive, 4),
            "exploit_drive": round(exploit_drive, 4),
            **{f"planner_{k}": v for k, v in planner_breakdown.items()},

            # All 7 action values (total)
            "val_NORTH": all_action_vals["MOVE NORTH"],
            "val_SOUTH": all_action_vals["MOVE SOUTH"],
            "val_EAST": all_action_vals["MOVE EAST"],
            "val_WEST": all_action_vals["MOVE WEST"],
            "val_WAIT": all_action_vals["WAIT"],
            "val_DROP": all_action_vals["DROP ENERGY"],
            "val_WALL": all_action_vals["BUILD WALL"],
            "val_BEST": max(all_action_vals.values()),
            "val_BEST_ACTION": max(all_action_vals, key=all_action_vals.get),
            "val_SPREAD": round(max(all_action_vals.values()) - min(all_action_vals.values()), 4),

            # Full component breakdown for each action (7 actions x 10 fields = 70 fields)
            "north_pragmatic": all_action_breakdowns["move_north"]["pragmatic"],
            "north_threat": all_action_breakdowns["move_north"]["threat_penalty"],
            "north_battery": all_action_breakdowns["move_north"]["battery_penalty"],
            "north_wait": all_action_breakdowns["move_north"]["wait_penalty"],
            "north_commit": all_action_breakdowns["move_north"]["commitment_penalty"],
            "north_explore": all_action_breakdowns["move_north"]["exploration_bonus"],
            "north_social": all_action_breakdowns["move_north"]["social_value"],
            "north_empathy": all_action_breakdowns["move_north"]["empathy_trauma"],
            "north_wall_bonus": all_action_breakdowns["move_north"]["wall_bonus"],
            "north_sim_ob": all_action_breakdowns["move_north"]["sim_other_battery"],

            "south_pragmatic": all_action_breakdowns["move_south"]["pragmatic"],
            "south_threat": all_action_breakdowns["move_south"]["threat_penalty"],
            "south_battery": all_action_breakdowns["move_south"]["battery_penalty"],
            "south_wait": all_action_breakdowns["move_south"]["wait_penalty"],
            "south_commit": all_action_breakdowns["move_south"]["commitment_penalty"],
            "south_explore": all_action_breakdowns["move_south"]["exploration_bonus"],
            "south_social": all_action_breakdowns["move_south"]["social_value"],
            "south_empathy": all_action_breakdowns["move_south"]["empathy_trauma"],
            "south_wall_bonus": all_action_breakdowns["move_south"]["wall_bonus"],
            "south_sim_ob": all_action_breakdowns["move_south"]["sim_other_battery"],

            "east_pragmatic": all_action_breakdowns["move_east"]["pragmatic"],
            "east_threat": all_action_breakdowns["move_east"]["threat_penalty"],
            "east_battery": all_action_breakdowns["move_east"]["battery_penalty"],
            "east_wait": all_action_breakdowns["move_east"]["wait_penalty"],
            "east_commit": all_action_breakdowns["move_east"]["commitment_penalty"],
            "east_explore": all_action_breakdowns["move_east"]["exploration_bonus"],
            "east_social": all_action_breakdowns["move_east"]["social_value"],
            "east_empathy": all_action_breakdowns["move_east"]["empathy_trauma"],
            "east_wall_bonus": all_action_breakdowns["move_east"]["wall_bonus"],
            "east_sim_ob": all_action_breakdowns["move_east"]["sim_other_battery"],

            "west_pragmatic": all_action_breakdowns["move_west"]["pragmatic"],
            "west_threat": all_action_breakdowns["move_west"]["threat_penalty"],
            "west_battery": all_action_breakdowns["move_west"]["battery_penalty"],
            "west_wait": all_action_breakdowns["move_west"]["wait_penalty"],
            "west_commit": all_action_breakdowns["move_west"]["commitment_penalty"],
            "west_explore": all_action_breakdowns["move_west"]["exploration_bonus"],
            "west_social": all_action_breakdowns["move_west"]["social_value"],
            "west_empathy": all_action_breakdowns["move_west"]["empathy_trauma"],
            "west_wall_bonus": all_action_breakdowns["move_west"]["wall_bonus"],
            "west_sim_ob": all_action_breakdowns["move_west"]["sim_other_battery"],

            "wait_pragmatic": all_action_breakdowns["wait"]["pragmatic"],
            "wait_threat": all_action_breakdowns["wait"]["threat_penalty"],
            "wait_battery": all_action_breakdowns["wait"]["battery_penalty"],
            "wait_wait": all_action_breakdowns["wait"]["wait_penalty"],
            "wait_commit": all_action_breakdowns["wait"]["commitment_penalty"],
            "wait_explore": all_action_breakdowns["wait"]["exploration_bonus"],
            "wait_social": all_action_breakdowns["wait"]["social_value"],
            "wait_empathy": all_action_breakdowns["wait"]["empathy_trauma"],
            "wait_wall_bonus": all_action_breakdowns["wait"]["wall_bonus"],
            "wait_sim_ob": all_action_breakdowns["wait"]["sim_other_battery"],

            "drop_pragmatic": all_action_breakdowns["drop_energy"]["pragmatic"],
            "drop_threat": all_action_breakdowns["drop_energy"]["threat_penalty"],
            "drop_battery": all_action_breakdowns["drop_energy"]["battery_penalty"],
            "drop_wait": all_action_breakdowns["drop_energy"]["wait_penalty"],
            "drop_commit": all_action_breakdowns["drop_energy"]["commitment_penalty"],
            "drop_explore": all_action_breakdowns["drop_energy"]["exploration_bonus"],
            "drop_social": all_action_breakdowns["drop_energy"]["social_value"],
            "drop_empathy": all_action_breakdowns["drop_energy"]["empathy_trauma"],
            "drop_wall_bonus": all_action_breakdowns["drop_energy"]["wall_bonus"],
            "drop_sim_ob": all_action_breakdowns["drop_energy"]["sim_other_battery"],
            "drop_safety_margin": all_action_breakdowns["drop_energy"]["safety_margin"],

            "wall_pragmatic": all_action_breakdowns["build_wall"]["pragmatic"],
            "wall_threat": all_action_breakdowns["build_wall"]["threat_penalty"],
            "wall_battery": all_action_breakdowns["build_wall"]["battery_penalty"],
            "wall_wait": all_action_breakdowns["build_wall"]["wait_penalty"],
            "wall_commit": all_action_breakdowns["build_wall"]["commitment_penalty"],
            "wall_explore": all_action_breakdowns["build_wall"]["exploration_bonus"],
            "wall_social": all_action_breakdowns["build_wall"]["social_value"],
            "wall_empathy": all_action_breakdowns["build_wall"]["empathy_trauma"],
            "wall_wall_bonus": all_action_breakdowns["build_wall"]["wall_bonus"],
            "wall_sim_ob": all_action_breakdowns["build_wall"]["sim_other_battery"],

            # Component-preference analysis
            "best_pragmatic_action": best_pragmatic_action,
            "best_social_action": best_social_action,
            "best_empathy_action": best_empathy_action,

            # Social condition detail
            **social_detail,

            # --- GENOME (full state this cycle) ---
            **genome_detail,

            # --- HOMEOSTATIC FITNESS TRACKING ---
            **homeo_stats,
            "homeo_fitness_now": round(ica.evaluate_homeostatic_fitness(ica.cycle_count, ica.agent_stats), 4),

            # --- FORWARD MODEL: PRE (9-dim) ---
            "pre_dx_energy": round(raw_state_pre[0].item(), 4),
            "pre_dy_energy": round(raw_state_pre[1].item(), 4),
            "pre_dx_threat": round(raw_state_pre[2].item(), 4),
            "pre_dy_threat": round(raw_state_pre[3].item(), 4),
            "pre_dx_other": round(raw_state_pre[4].item(), 4),
            "pre_dy_other": round(raw_state_pre[5].item(), 4),
            "pre_battery": round(raw_state_pre[6].item(), 4),
            "pre_health": round(raw_state_pre[7].item(), 4),
            "pre_other_battery": round(raw_state_pre[8].item(), 4),

            # --- FORWARD MODEL: PREDICTED (9-dim) ---
            "pred_dx_energy": round(predicted_next_state[0].item(), 4),
            "pred_dy_energy": round(predicted_next_state[1].item(), 4),
            "pred_dx_threat": round(predicted_next_state[2].item(), 4),
            "pred_dy_threat": round(predicted_next_state[3].item(), 4),
            "pred_dx_other": round(predicted_next_state[4].item(), 4),
            "pred_dy_other": round(predicted_next_state[5].item(), 4),
            "pred_battery": round(predicted_next_state[6].item(), 4),
            "pred_health": round(predicted_next_state[7].item(), 4),
            "pred_other_battery": round(predicted_next_state[8].item(), 4),

            # --- FORWARD MODEL: POST (9-dim) ---
            "post_dx_energy": round(raw_state_post[0].item(), 4),
            "post_dy_energy": round(raw_state_post[1].item(), 4),
            "post_dx_threat": round(raw_state_post[2].item(), 4),
            "post_dy_threat": round(raw_state_post[3].item(), 4),
            "post_dx_other": round(raw_state_post[4].item(), 4),
            "post_dy_other": round(raw_state_post[5].item(), 4),
            "post_battery": round(raw_state_post[6].item(), 4),
            "post_health": round(raw_state_post[7].item(), 4),
            "post_other_battery": round(raw_state_post[8].item(), 4),

            # --- PREDICTION ACCURACY (per-dim) ---
            "pe_dx_energy": round(abs(predicted_next_state[0].item() - raw_state_post[0].item()), 6),
            "pe_dy_energy": round(abs(predicted_next_state[1].item() - raw_state_post[1].item()), 6),
            "pe_dx_threat": round(abs(predicted_next_state[2].item() - raw_state_post[2].item()), 6),
            "pe_dy_threat": round(abs(predicted_next_state[3].item() - raw_state_post[3].item()), 6),
            "pe_dx_other": round(abs(predicted_next_state[4].item() - raw_state_post[4].item()), 6),
            "pe_dy_other": round(abs(predicted_next_state[5].item() - raw_state_post[5].item()), 6),
            "pe_battery": round(abs(predicted_next_state[6].item() - raw_state_post[6].item()), 6),
            "pe_health": round(abs(predicted_next_state[7].item() - raw_state_post[7].item()), 6),
            "pe_other_battery": round(abs(predicted_next_state[8].item() - raw_state_post[8].item()), 6),

            # --- LEARNING SIGNALS ---
            "prediction_error": round(prediction_error, 6),
            "curiosity": round(intrinsic_curiosity, 6),
            "rpe": round(rpe, 6),
            "rpe_pred_before": round(pred_val_before, 6),
            "rpe_joint_reward": round(joint_reward, 6),
            "reward_raw": reward,
            "pain": pain,
            "social_reward": round(social_reward, 4),
            "neuromod": round(neuromod, 6),
            "altruism": altruism,
            "wall_build_action": wall_build,
            "valence": round(valence, 4),

            # --- META-PLASTICITY ---
            "meta_lambda": round(dynamic_lambda, 6),
            "dynamic_lr": round(dynamic_lr, 6),
            "surprise_window_len": len(ica.meta_controller.error_window),
            "surprise_window_avg": round(sum(ica.meta_controller.error_window) / len(ica.meta_controller.error_window), 6) if ica.meta_controller.error_window else None,
            "rigidity_factor": round(min(max(1.0 / (1.0 + (sum(ica.meta_controller.error_window) / len(ica.meta_controller.error_window)) * 5.0), 0.1), 1.0), 6) if ica.meta_controller.error_window else None,

            # --- DYNAMIC WORLD ---
            "entropy_cycle": world.entropy_counter,
            "entropy_event": world.entropy_counter > 0 and world.entropy_counter % 30 == 0,
            "physics_inverted": world.physics_inverted,
            "num_energy_tiles": len(world.energy_tiles),
            "energy_tiles": world.energy_tiles,
            "daytime": world.time_of_day < 50,
            "grid_size": world.grid_size,

            # --- COGNITIVE MAP ---
            "visited_cells": visited_cells,
            "unvisited_cells": total_cells - visited_cells,
            "exploration_pct": round(visited_cells / total_cells * 100, 1) if total_cells else 0.0,
            "cogmap_energy_belief_max": round(float(ica.cognitive_map.map[0].max().item()), 4),
            "cogmap_threat_belief_max": round(float(ica.cognitive_map.map[1].max().item()), 4),
            "cogmap_other_belief_max": round(float(ica.cognitive_map.map[2].max().item()), 4),
            "cogmap_energy_entropy": round(float((-ica.cognitive_map.map[0] * torch.log(ica.cognitive_map.map[0] + 1e-8)).sum().item()), 4),
            "cogmap_threat_entropy": round(float((-ica.cognitive_map.map[1] * torch.log(ica.cognitive_map.map[1] + 1e-8)).sum().item()), 4),
            "cogmap_other_entropy": round(float((-ica.cognitive_map.map[2] * torch.log(ica.cognitive_map.map[2] + 1e-8)).sum().item()), 4),
            "cogmap_grid_visited": visited_str,
            **cogmap_grids,

            # --- GLOBAL WORKSPACE ---
            "phi": round(ica.hom.phi, 6),
            "inhibition_signal": round(ica.hom.inhibition_signal, 6),
            "latent_vibe_norm": round(ica.workspace.latent_state.norm().item(), 6),
            "attention_target": ica.hom.attention_target or "none",
            "workspace_broadcast": workspace_text[:200],
            **ws_detail,

            # --- OTHER AGENT ---
            "other_x": other.x,
            "other_y": other.y,
            "other_battery": round(other.battery, 1),
            "other_starving": other.is_starving,
            "other_died": other_died,
            "is_attacked": is_attacked,
            "other_state": other.state,
            "social_collision": social_collision,
            "other_dist": abs(world.x - other.x) + abs(world.y - other.y),
            "other_dx": world.x - other.x,
            "other_dy": world.y - other.y,
            "other_health": other.health,

            # --- CORTICAL MODULES ---
            **{f"mod_{k}": v for k, v in module_activations.items()},
            **module_outputs,

            # --- PIPELINE ---
            "pipeline_total_ms": round(pipeline_latency.get("total_ms", 0), 2),
            "pipeline_stage_ms": {k: round(v, 2) for k, v in pipeline_latency.get("stage_ms", {}).items()},

            # --- MEMORY INDEX ---
            "memory_index_size": len(ica.memory_index.embeddings),
            "memory_index_recall_count": len(fast_recall) if fast_recall else 0,
            "memory_index_recall_top_sim": round(fast_recall[0][2], 4) if fast_recall else 0.0,

            # --- PROACTIVE MONITOR ---
            "proactive_message": proactive_msg or "",

            # --- EMOTION / SELF-NARRATIVE ---
            "emotion_text": emo[:80] if emo else "",
            "self_narrative": self_narr[:120] if self_narr else "",
            "emotion_temp": round(emotion_temp, 3),

            # --- MEMORY ---
            "hippocampal_buffer_len": len(ica.hippocampal_buffer),
            "episodic_memory_size": len(ica.episodic_memory.episodes),
            "experience_buffer_len": len(ica.experience_buffer),
            **hippo_stats,
            **episodic_snap,

            # --- SLEEP ---
            "micro_sleep": micro_sleep,
            "scheduled_sleep": scheduled_sleep,
            "did_sleep": did_sleep,
            "sleep_count": len(sleep_history),
            "neurogenesis": neurogenesis,

            # --- THREAT ---
            "threat_level": round(threat_level, 4),
            "survival_override": override is not None,

            # --- VALUE NET ---
            "prosocial_pred_value": round(pred_val_before, 6),
            "empathy_weight": 0.3,

            # --- CIRCADIAN ---
            **clock_details,

            # --- WEIGHT STATS ---
            **lora_stats,
            **lora_grads,
            **ewc_stats,

            # --- LIMBIC AFFECT (Phase 24) ---
            **affect_detail,

            # --- RUN METADATA ---
            "run": run_num,
            "wall_build_events_so_far": len(wall_build_events),
            "self_pres_vetoes_so_far": len(self_pres_veto_events),
            "altruism_attempts_so_far": len(altruism_attempts),
        }

        # Print one-line summary
        depth_str = f" d={planning_depth}" if planning_depth else ""
        src_tag = f" [{action_source}]" if action_source != "EXPLORE" else ""
        inv_tag = " INV!" if world.physics_inverted else ""
        social_tag = ""
        if social_detail["social_v7_can_drop"]:
            social_tag = " [CAN_DROP]"
        if action == "DROP ENERGY":
            social_tag = " [DROPPED]"
        if action == "BUILD WALL":
            social_tag = " [BUILD]"
        veto_tag = " [VETO]" if chosen_veto else ""
        empathy_tag = ""
        if social_detail["social_drop_saves_other"]:
            empathy_tag = " [EMPATHY]"
        wall_tag = f" w={len(world.walls)}" if wall_built_this_cycle or on_wall else ""

        print(f"  C{entry['cycle']:2d}{inv_tag} | Bat={entry['battery_pct']:5.1f}% HP={entry['health_pct']:5.1f}% | "
              f"{entry['action']:12s}{depth_str}{src_tag}{social_tag}{empathy_tag}{veto_tag}{wall_tag} | PE={entry['prediction_error']:.3f} "
              f"Cur={entry['curiosity']:.3f} | NM={entry['neuromod']:.2f} "
              f"V={visited_cells:2d}/{total_cells}"
               f" | O={entry['other_battery']:5.1f}%{' ATTACKED!' if is_attacked else ''}{' HOSTILE!' if entry.get('other_state') == 'Hostile' else ''}{' STARVING!' if entry['other_starving'] else ''}{' DIED!' if other_died else ''}"
               f" | {entry.get('affect_mood','?')} | EWCl={entry['genome_ewc_lambda']:.0f} SV={entry['genome_social_value']:.2f} BP={entry['genome_battery_penalty']:.2f}"
              f"{' | DEAD' if dead else ''}")

        log_entries.append(entry)

        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        # Also log genome evolution separately
        genome_entry = genome_detail.copy()
        genome_entry.update({
            "cycle": ica.cycle_count,
            "run": run_num,
            "steps_alive": world.steps_alive,
            "battery_pct": round(world.battery, 1),
            "homeo_fitness": round(ica.evaluate_homeostatic_fitness(ica.cycle_count, ica.agent_stats), 4),
        })
        genome_entries.append(genome_entry)
        with open(GENOME_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(genome_entry) + "\n")

        if dead:
            raw_homeo = ica.evaluate_homeostatic_fitness(ica.cycle_count, ica.agent_stats)
            fitness_for_ema = max(0.0, min(1.0, raw_homeo / 2.0))
            ica.genome.fitness = (ica.genome.fitness * 0.8) + (fitness_for_ema * 0.2)
            print(f"  >>> DIED at cycle {ica.cycle_count} ({world.steps_alive} steps | homeo_fitness={raw_homeo:.4f} ema_input={fitness_for_ema:.4f})")
            print(f"  [DEATH] stats={ica.agent_stats}")
            ica.genome.mutate(fitness_for_ema)
            ica.genome.save()
            ica.check_and_trigger_neurogenesis()
            ica.synaptic_turnover(turnover_rate=0.10)
            ica.meta_controller.base_lambda = ica.genome.genes['ewc_base_lambda']
            ica.meta_controller.base_lr = ica.genome.genes['learning_rate']
            print(f"  [GENOME] lr={ica.genome.genes['learning_rate']:.6f} ewc_lambda={ica.genome.genes['ewc_base_lambda']:.2f} "
                  f"cur={ica.genome.genes['curiosity_weight']:.2f} social={ica.genome.genes['social_value']:.2f} "
                  f"bat_pen={ica.genome.genes['battery_penalty']:.2f} depth={ica.genome.genes['planning_depth']}")
            death_count_this_life += 1
            ica.cognitive_map.map.zero_()
            ica.cognitive_map.visited.zero_()
            ica.world.reset()
            ica.workspace.clear()
            for m in ica.modules.values():
                m.activation = 0.0
                m.last_output = ""
            ica.agent_stats = {'energy_gained': 0, 'cells_explored': 0, 'threat_hits': 0, 'social_acts': 0, 'starving_cycles': 0}
            ica.cycle_count = 0
            break

    return log_entries, wall_build_events, self_pres_veto_events, altruism_attempts, genome_entries

def summarize_life(entries, wall_events, veto_events, altruism_attempts):
    if not entries:
        return {"error": "no_cycles"}
    pe_vals = [e["prediction_error"] for e in entries]
    cur_vals = [e["curiosity"] for e in entries]
    nm_vals = [e["neuromod"] for e in entries]
    deps = [e["planning_depth"] for e in entries if e["planning_depth"] is not None]
    repeats = [e["action_repeats"] for e in entries if e["action_repeats"] >= 2]
    altruism_cycle = [e["cycle"] for e in entries if e.get("altruism", False)]
    other_death_cycles = [e["cycle"] for e in entries if e.get("other_died", False)]
    wall_cycles = [e["cycle"] for e in entries if e.get("wall_built_this_cycle", False)]
    north_count = sum(1 for e in entries if e.get("val_NORTH", -999) >= max(e.get("val_SOUTH",-999), e.get("val_EAST",-999), e.get("val_WEST",-999), e.get("val_WAIT",-999)))
    drop_best_count = sum(1 for e in entries if e.get("val_DROP", -999) >= max(e.get("val_NORTH",-999), e.get("val_SOUTH",-999), e.get("val_EAST",-999), e.get("val_WEST",-999), e.get("val_WAIT",-999)))
    wall_best_count = sum(1 for e in entries if e.get("val_WALL", -999) >= max(e.get("val_NORTH",-999), e.get("val_SOUTH",-999), e.get("val_EAST",-999), e.get("val_WEST",-999), e.get("val_WAIT",-999)))
    social_conditions_met = sum(1 for e in entries if e.get("social_v7_can_drop", False))
    social_adjacent_starving = sum(1 for e in entries if e.get("social_adjacent_and_starving", False))
    veto_count = len([e for e in entries if e.get("self_pres_veto_hit", False)])

    # Track initial and final genome
    first_g = {k: entries[0].get(f"genome_{k}", None) for k in ["ewc_lambda","social_value","battery_penalty","planning_depth","lr","curiosity","fitness"]}
    last_g = {k: entries[-1].get(f"genome_{k}", None) for k in ["ewc_lambda","social_value","battery_penalty","planning_depth","lr","curiosity","fitness"]}

    return {
        "cycles_lived": len(entries),
        "battery_start": entries[0]["battery_pct"],
        "battery_end": entries[-1]["battery_pct"],
        "health_end": entries[-1]["health_pct"],
        "death_cause": "battery" if entries[-1]["battery_pct"] <= 0 else "health" if entries[-1]["health_pct"] <= 0 else "survived",
        "avg_PE": round(sum(pe_vals) / len(pe_vals), 6),
        "avg_Curiosity": round(sum(cur_vals) / len(cur_vals), 6),
        "avg_Neuromod": round(sum(nm_vals) / len(nm_vals), 6),
        "depth_use": {d: deps.count(d) for d in sorted(set(deps))},
        "total_reward": sum(e["reward_raw"] for e in entries),
        "total_pain": sum(e["pain"] for e in entries),
        "total_social_reward": sum(e["social_reward"] for e in entries),
        "avg_phi": round(sum(e["phi"] for e in entries) / len(entries), 6),
        "action_repeat_events": len(repeats),
        "max_action_repeats": max(repeats) if repeats else 0,
        "final_visited_cells": int(entries[-1].get("visited_cells", 0)),
        "exploration_pct": round(entries[-1].get("visited_cells", 0) / 49 * 100, 1),
        "social_collisions": sum(e["social_collision"] for e in entries),
        "episodic_stores": entries[-1].get("episodic_memory_size", 0),
        "altruism_acts": sum(1 for e in entries if e.get("altruism", False)),
        "altruism_at_cycles": altruism_cycle,
        "drop_best_count": drop_best_count,
        "drop_best_pct": round(drop_best_count / len(entries) * 100, 1) if entries else 0.0,
        "social_conditions_met": social_conditions_met,
        "social_adjacent_starving": social_adjacent_starving,
        "other_deaths": sum(1 for e in entries if e.get("other_died", False)),
        "other_death_at_cycles": other_death_cycles,
        "entropy_shifts": sum(1 for e in entries if e.get("entropy_cycle", 0) % 30 == 0),
        "final_energy_tiles": entries[-1].get("num_energy_tiles", 1),
        "avg_val_spread": round(sum(e.get("val_SPREAD", 0) for e in entries) / len(entries), 4),
        "north_advantage": round(north_count / len(entries), 3),

        # Phase 22 specific
        "wall_builds": len(wall_events),
        "wall_at_cycles": wall_cycles,
        "wall_best_count": wall_best_count,
        "wall_best_pct": round(wall_best_count / len(entries) * 100, 1) if entries else 0.0,
        "self_pres_veto_events": len(veto_events),
        "self_pres_veto_cycles": [v["cycle"] for v in veto_events],
        "self_pres_veto_reasons": [v["veto_reason"] for v in veto_events],
        "altruism_attempts": len(altruism_attempts),
        "altruism_survivable": sum(1 for a in altruism_attempts if a.get("survivable", False)),
        "altruism_nonsurvivable": sum(1 for a in altruism_attempts if not a.get("survivable", False)),
        "num_walls_end": entries[-1].get("num_walls", 0),

        # Gene tracking
        "genome_ewc_lambda_start": first_g.get("ewc_lambda"),
        "genome_ewc_lambda_end": last_g.get("ewc_lambda"),
        "genome_social_value_start": first_g.get("social_value"),
        "genome_social_value_end": last_g.get("social_value"),
        "genome_battery_penalty_start": first_g.get("battery_penalty"),
        "genome_battery_penalty_end": last_g.get("battery_penalty"),
        "genome_fitness_start": first_g.get("fitness"),
        "genome_fitness_end": last_g.get("fitness"),
    }

def run_diagnostic():
    results = {}
    all_wall_events = []
    all_veto_events = []
    all_altruism_attempts = []
    all_genome_entries = []

    for run in range(RUNS):
        print(f"\n{'='*60}")
        print(f"RUN {run+1}/{RUNS} (seed={SEED+run}) — Phase 22 Self-Preservation Imperative")
        print(f"{'='*60}")
        torch.manual_seed(SEED + run)
        random.seed(SEED + run)
        ica.cognitive_map.map.zero_()
        ica.cognitive_map.visited.zero_()
        ica.world.reset()
        ica.workspace.clear()
        for m in ica.modules.values():
            m.activation = 0.0
            m.last_output = ""
        ica.cycle_count = 0
        ica.last_real_action = "WAIT"

        live_data, wall_events, veto_events, altruism_attempts, genome_entries = run_life(MAX_CYCLES, run + 1)
        results[f"run_{run+1}"] = live_data
        all_wall_events.extend(wall_events)
        all_veto_events.extend(veto_events)
        all_altruism_attempts.extend(altruism_attempts)
        all_genome_entries.extend(genome_entries)

        summary = summarize_life(live_data, wall_events, veto_events, altruism_attempts)
        print(f"\n  RUN {run+1} SUMMARY:")
        for k, v in summary.items():
            print(f"    {k}: {v}")

    return results, all_wall_events, all_veto_events, all_altruism_attempts, all_genome_entries

if __name__ == "__main__":
    for log_path in [LOG, GENOME_LOG]:
        if os.path.exists(log_path):
            os.remove(log_path)

    print(f"ICA Phase 22 Diagnostic — Self-Preservation Imperative, EWC Clamping, Tuned Fitness")
    print(f"{RUNS} runs, {MAX_CYCLES} max cycles each")
    print(f"Model: {ica.model.__class__.__name__} ({sum(p.numel() for p in ica.model.parameters())} params)")
    start = time.time()
    results, all_wall_events, all_veto_events, all_altruism_attempts, all_genome_entries = run_diagnostic()
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print("FINAL AGGREGATED SUMMARY — PHASE 22")
    print(f"{'='*60}")
    all_lifespans = []
    all_pe = []
    all_cur = []
    all_nm = []
    all_visited = []
    total_cycles = 0
    total_altruism = 0
    total_other_deaths = 0
    total_social_collisions = 0
    total_action_repeats = 0
    total_drop_best = 0
    total_wall_best = 0
    total_wall_builds = 0
    total_vetoes = 0
    total_altruism_attempts = 0
    total_altruism_nonsurvivable = 0
    ewc_lambdas_end = []
    bat_pens_end = []
    social_vals_end = []

    for rname, entries in results.items():
        s = summarize_life(entries, [], [], [])
        all_lifespans.append(s["cycles_lived"])
        all_pe.extend(e["prediction_error"] for e in entries)
        all_cur.extend(e["curiosity"] for e in entries)
        all_nm.extend(e["neuromod"] for e in entries)
        all_visited.append(s["final_visited_cells"])
        total_altruism += s["altruism_acts"]
        total_other_deaths += s["other_deaths"]
        total_social_collisions += s["social_collisions"]
        total_action_repeats += s["action_repeat_events"]
        total_drop_best += s["drop_best_count"]
        total_wall_best += s["wall_best_count"]
        total_wall_builds += s["wall_builds"]
        total_vetoes += s["self_pres_veto_events"]
        total_altruism_attempts += s["altruism_attempts"]
        total_altruism_nonsurvivable += s["altruism_nonsurvivable"]
        if s["genome_ewc_lambda_end"] is not None:
            ewc_lambdas_end.append(s["genome_ewc_lambda_end"])
        if s["genome_battery_penalty_end"] is not None:
            bat_pens_end.append(s["genome_battery_penalty_end"])
        if s["genome_social_value_end"] is not None:
            social_vals_end.append(s["genome_social_value_end"])
        total_cycles += s["cycles_lived"]

    print(f"  Lives: {RUNS}")
    print(f"  Total cycles lived: {total_cycles}")
    print(f"  Avg lifespan: {sum(all_lifespans)/len(all_lifespans):.1f} cycles")
    print(f"  Avg PE: {sum(all_pe)/len(all_pe):.6f}")
    print(f"  Avg Curiosity: {sum(all_cur)/len(all_cur):.6f}")
    print(f"  Avg Neuromod: {sum(all_nm)/len(all_nm):.6f}")

    print(f"\n  --- GENETIC INTEGRITY (EWC Clamping) ---")
    print(f"  EWC lambda end values: {ewc_lambdas_end}")
    print(f"  EWC min observed: {min(ewc_lambdas_end) if ewc_lambdas_end else 'N/A'}")
    print(f"  EWC clamping PASSED (>= 500): {all(v >= 500 for v in ewc_lambdas_end) if ewc_lambdas_end else 'N/A'}")

    print(f"\n  --- SELF-PRESERVATION IMPERATIVE ---")
    print(f"  Total veto events: {total_vetoes}")
    print(f"  Veto details by cycle:")
    for v in all_veto_events:
        print(f"    C{v['cycle']}: {v['veto_reason']} (bat={v['battery']}%, threat_dist={v['threat_dist']})")
    print(f"  Zero vetos = self-preservation not needed: {total_vetoes == 0}")
    print(f"  Non-zero vetos = AI prevented from suicide: {total_vetoes > 0}")

    print(f"\n  --- NICHE CONSTRUCTION (BUILD WALL) ---")
    print(f"  Total wall builds: {total_wall_builds}")
    print(f"  Walls per run: {[s['wall_builds'] for s in [summarize_life(v, [], [], []) for v in results.values()]]}")
    print(f"  Wall events detail:")
    for w in all_wall_events:
        print(f"    C{w['cycle']}: ({w['x']},{w['y']}) bat={w['battery_before']}->{w['battery_after']}% threat_dist={w['threat_dist']} wall_val={w['wall']} wall_bonus={w['wall_bonus']}")
    wall_below_20 = [w for w in all_wall_events if w['battery_before'] <= 20]
    print(f"  BUILD WALL while <= 20% battery: {len(wall_below_20)} {'❌ FAIL' if wall_below_20 else '✅ PASS'}")

    print(f"\n  --- ALTRUISM SAFETY ---")
    print(f"  Total altruism acts: {total_altruism}")
    print(f"  Total altruism attempts (incl vetoed): {total_altruism_attempts}")
    print(f"  Non-survivable attempts (self-pres would veto): {total_altruism_nonsurvivable}")
    for a in all_altruism_attempts:
        tag = " SURVIVABLE" if a.get("survivable") else " NON-SURVIVABLE"
        print(f"    C{a['cycle']}: bat={a['battery_before']}% -> {a['battery_after']}% other={a['other_battery_before']}% dist={a['other_dist']} safety_margin={a.get('safety_margin',0):.4f}{tag}")

    print(f"\n  --- BATTERY PENALTY CLAMPING ---")
    print(f"  Battery penalty end values: {bat_pens_end}")
    print(f"  Clamping PASSED (<= -0.1): {all(v <= -0.1 for v in bat_pens_end) if bat_pens_end else 'N/A'}")

    print(f"\n  --- SOCIAL VALUE CLAMPING ---")
    print(f"  Social value end values: {social_vals_end}")
    print(f"  Clamping PASSED (>= 0): {all(v >= 0 for v in social_vals_end) if social_vals_end else 'N/A'}")

    print(f"\n  --- OTHER METRICS ---")
    print(f"  Total altruism acts: {total_altruism}")
    print(f"  Total other-agent deaths: {total_other_deaths}")
    print(f"  Times DROP was best: {total_drop_best}")
    print(f"  Times BUILD WALL was best: {total_wall_best}")
    print(f"  Avg visited cells: {sum(all_visited)/len(all_visited):.1f}/49")
    print(f"  Total social collisions: {total_social_collisions}")
    print(f"  Total action repeat events: {total_action_repeats}")
    print(f"  Time elapsed: {elapsed:.1f}s")

    print(f"\n  Full detailed log: {LOG}")
    print(f"  Genome evolution log: {GENOME_LOG}")
    print(f"  ~{len(all_genome_entries)} genome snapshots written")
