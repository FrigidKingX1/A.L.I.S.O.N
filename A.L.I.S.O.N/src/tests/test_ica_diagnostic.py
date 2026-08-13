"""Highly detailed diagnostic test for ICA agent — v3 with ~200 fields."""
import os, sys, json, time, random, math
import torch

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import alison_core as ica

LOG = "ica_diagnostic_log.jsonl"
RUNS = 3
MAX_CYCLES = 60
SEED = 42

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
    """Gradient norms per LoRA module — shows what's actually learning."""
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

def compute_all_action_breakdowns(world, sfm, raw_state, cog_map):
    """Returns all 7+1 components of evaluate_continuous_state_v7 for each of 6 actions (includes empathy_trauma)."""
    actions = ["MOVE NORTH", "MOVE SOUTH", "MOVE EAST", "MOVE WEST", "WAIT", "DROP ENERGY"]
    last = ica.last_real_action
    other = ica.other_agent
    result = {}
    with torch.no_grad():
        for action in actions:
            if action == "DROP ENERGY":
                pred = raw_state.clone()
                pred[6] = max(0.0, pred[6].item() - 0.2)
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
            explor = 0.0
            if "NORTH" in action and world.y > 0:
                explor = 0.3 * (1.0 - cog_map.visited[world.y - 1, world.x].item())
            elif "SOUTH" in action and world.y < 6:
                explor = 0.3 * (1.0 - cog_map.visited[world.y + 1, world.x].item())
            elif "EAST" in action and world.x < 6:
                explor = 0.3 * (1.0 - cog_map.visited[world.y, world.x + 1].item())
            elif "WEST" in action and world.x > 0:
                explor = 0.3 * (1.0 - cog_map.visited[world.y, world.x - 1].item())

            # Compute simulated other battery after this action (as the planner does at depth 0)
            dto = abs(world.x - other.x) + abs(world.y - other.y)
            if action == "DROP ENERGY":
                if dto <= 1:
                    sim_ob = min(100.0, other.battery + 30.0)
                else:
                    sim_ob = other.battery - 3.0
            else:
                sim_ob = other.battery - 3.0

            social = 0.0
            if action == "DROP ENERGY":
                if world.battery > 40 and dto <= 1:
                    social = ica.genome.genes['social_value']
                else:
                    social = -2.0

            empathy_trauma = -5.0 * ica.genome.genes['social_value'] if sim_ob <= 0 else 0.0

            total = round(pragmatic + threat + bat_pen + wait_pen + commit + explor + social + empathy_trauma, 4)
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
                "sim_other_battery": round(sim_ob, 1),
                "total": total,
            }
    return result

def get_social_condition_detail():
    """Returns the v7 condition checks for altruism (simulated empathy)."""
    other = ica.other_agent
    world = ica.world
    dto = abs(world.x - other.x) + abs(world.y - other.y)
    sim_ob_no_drop = other.battery - 3.0
    sim_ob_drop = min(100.0, other.battery + 30.0) if dto <= 1 else other.battery - 3.0
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
    }

def get_cogmap_grids(cog_map):
    return {
        "cogmap_energy_grid": [round(float(cog_map.map[0, r, c].item()), 4) for r in range(7) for c in range(7)],
        "cogmap_threat_grid": [round(float(cog_map.map[1, r, c].item()), 4) for r in range(7) for c in range(7)],
        "cogmap_other_grid": [round(float(cog_map.map[2, r, c].item()), 4) for r in range(7) for c in range(7)],
        "cogmap_visited_grid": [int(cog_map.visited[r, c].item()) for r in range(7) for c in range(7)],
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
    exploration_bonus = 0.0
    if "NORTH" in action and world.y > 0:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y - 1, world.x].item())
    elif "SOUTH" in action and world.y < 6:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y + 1, world.x].item())
    elif "EAST" in action and world.x < 6:
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

def run_diagnostic():
    results = {}
    for run in range(RUNS):
        print(f"\n{'='*60}")
        print(f"RUN {run+1}/{RUNS} (seed={SEED+run})")
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

        live_data = run_life(MAX_CYCLES)
        results[f"run_{run+1}"] = live_data
        summary = summarize_life(live_data)
        print(f"\n  RUN {run+1} SUMMARY:")
        for k, v in summary.items():
            print(f"    {k}: {v}")
    return results

def run_life(max_cycles):
    log_entries = []
    prev_action = None
    consecutive_count = 0
    planner_vals_history = []
    sleep_history = []

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

        # -- PHASE 2: CORTICAL PIPELINE (Metacognition) --
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
        action_source = "EXPLORE"

        # Compute full component breakdown for ALL 6 actions (for deep analysis)
        all_action_breakdowns = compute_all_action_breakdowns(world, ica.sensory_forward_model, raw_state_pre, ica.cognitive_map)

        # Also compute simple action values
        all_action_vals = {}
        for action in ["MOVE NORTH", "MOVE SOUTH", "MOVE EAST", "MOVE WEST", "WAIT", "DROP ENERGY"]:
            key = action.replace(" ", "_").lower()
            all_action_vals[action] = all_action_breakdowns[key]["total"]

        # Social condition detail
        social_detail = get_social_condition_detail()

        if ica.clock.state == "EXPLOITATION":
            action, planned_value = ica.adaptive_deep_planning_v5(world, ica.sensory_forward_model, raw_state_pre, ica.last_real_action, ica.cognitive_map, other)
            planning_depth = 5 if world.battery < 30 else (4 if world.battery < 60 else 2)
            altruism = (action == "DROP ENERGY")
            action_source = "EMERGENT_ALTRUISM" if altruism else "EMPATHETIC_IMAGINATION_V5"
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

        # -- PHASE 3: SELF-MODEL --
        self_narr = ica.hom.observe(ica.modules, ica.workspace.broadcast, threat_level)
        ica.workspace.consolidate(ica.model)

        workspace_text = ica.workspace.broadcast

        # -- PHASE 4: ACT --
        social_reward = 0.0
        other_died = False
        energy_found_this_cycle = False
        if action == "DROP ENERGY":
            social_reward = ica.execute_drop_energy(world, other)
            new_obs = world.get_observation()
            action_idx = ica.action_to_idx["WAIT"]
            predicted_next_state = ica.sensory_forward_model.predict_next_state(raw_state_pre, action_idx)
            ica.curiosity_module.set_prediction(predicted_next_state)
            reward, dead, pain = 0, False, 0
        else:
            action_idx = ica.action_to_idx.get(action, 4)
            predicted_next_state = ica.sensory_forward_model.predict_next_state(raw_state_pre, action_idx)
            ica.curiosity_module.set_prediction(predicted_next_state)
            new_obs, reward, dead, pain = world.step(action)

        # Check if agent found energy
        if world.x == world.energy[0] and world.y == world.energy[1]:
            energy_found_this_cycle = True
        if world.battery > raw_state_pre[6].item() + 10:
            energy_found_this_cycle = True

        other_status, stolen_battery = other.step(world, world.battery)
        if other_status == "DEAD":
            other_died = True
            other.reset()

        exp = {"prompt": f"Obs: {raw_obs} | Emo: {emo} | Act: {action}", "response": f" Result: {new_obs}"}
        ica.experience_buffer.append(exp)
        if len(ica.experience_buffer) > 20:
            ica.experience_buffer.pop(0)
        ica.latent_memory.update(ica.model, ica.workspace.broadcast)
        social_collision = (world.x == other.x and world.y == other.y)

        # -- PHASE 4c: PROACTIVE MONITOR --
        ica.proactive_monitor.check(world, other, ica.cognitive_map, cycle)
        proactive_msg = ica.proactive_monitor.proactive_message
        ica.proactive_monitor.proactive_message = None

        # -- PHASE 5: PREDICTION ERROR --
        raw_state_post = ica.get_raw_state(world)
        prediction_error = ica.sensory_forward_model.calculate_latent_fe(raw_state_pre, action_idx, raw_state_post)
        intrinsic_curiosity = ica.curiosity_module.calculate_curiosity(raw_state_post)
        dynamic_lambda, dynamic_lr = ica.meta_controller.update(prediction_error)

        # Prosocial ValueNet
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

        # -- GATHER ADDITIONAL DETAILS --
        lora_stats = get_lora_weight_stats()
        lora_grads = get_lora_grad_stats()
        ewc_stats = get_ewc_stats()
        cogmap_grids = get_cogmap_grids(ica.cognitive_map)
        episodic_snap = get_episodic_snapshot()
        hippo_stats = get_hippocampal_stats()
        clock_details = get_clock_details()
        module_outputs = get_module_outputs()
        ws_detail = get_workspace_detail()

        visited_str = "".join(str(int(ica.cognitive_map.visited[r, c].item())) for r in range(7) for c in range(7))

        # Determine which action would be best if we only considered each component
        best_pragmatic_action = max(all_action_breakdowns, key=lambda k: all_action_breakdowns[k]["pragmatic"])
        best_social_action = max(all_action_breakdowns, key=lambda k: all_action_breakdowns[k]["social_value"])
        best_empathy_action = max(all_action_breakdowns, key=lambda k: all_action_breakdowns[k]["empathy_trauma"])

        # -- BUILD HIGHLY DETAILED LOG ENTRY (~200 fields) --
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

            # --- ACTION ---
            "action": action,
            "action_idx": action_idx,
            "action_repeats": consecutive_count,
            "prev_action": prev_action if prev_action != action else "same",
            "action_source": action_source,

            # --- PLANNER ---
            "planning_depth": planning_depth,
            "planned_value": round(planned_value, 4) if planned_value is not None else None,
            "clock_state": ica.clock.state,
            "explore_drive": round(explore_drive, 4),
            "exploit_drive": round(exploit_drive, 4),
            **{f"planner_{k}": v for k, v in planner_breakdown.items()},

            # All 6 action values (total)
            "val_NORTH": all_action_vals["MOVE NORTH"],
            "val_SOUTH": all_action_vals["MOVE SOUTH"],
            "val_EAST": all_action_vals["MOVE EAST"],
            "val_WEST": all_action_vals["MOVE WEST"],
            "val_WAIT": all_action_vals["WAIT"],
            "val_DROP": all_action_vals["DROP ENERGY"],
            "val_BEST": max(all_action_vals.values()),
            "val_BEST_ACTION": max(all_action_vals, key=all_action_vals.get),
            "val_SPREAD": round(max(all_action_vals.values()) - min(all_action_vals.values()), 4),

            # Full component breakdown for each action (6 actions x 9 fields = 54 fields)
            # MOVE NORTH
            "north_pragmatic": all_action_breakdowns["move_north"]["pragmatic"],
            "north_threat": all_action_breakdowns["move_north"]["threat_penalty"],
            "north_battery": all_action_breakdowns["move_north"]["battery_penalty"],
            "north_wait": all_action_breakdowns["move_north"]["wait_penalty"],
            "north_commit": all_action_breakdowns["move_north"]["commitment_penalty"],
            "north_explore": all_action_breakdowns["move_north"]["exploration_bonus"],
            "north_social": all_action_breakdowns["move_north"]["social_value"],
            "north_empathy": all_action_breakdowns["move_north"]["empathy_trauma"],
            "north_sim_ob": all_action_breakdowns["move_north"]["sim_other_battery"],
            # MOVE SOUTH
            "south_pragmatic": all_action_breakdowns["move_south"]["pragmatic"],
            "south_threat": all_action_breakdowns["move_south"]["threat_penalty"],
            "south_battery": all_action_breakdowns["move_south"]["battery_penalty"],
            "south_wait": all_action_breakdowns["move_south"]["wait_penalty"],
            "south_commit": all_action_breakdowns["move_south"]["commitment_penalty"],
            "south_explore": all_action_breakdowns["move_south"]["exploration_bonus"],
            "south_social": all_action_breakdowns["move_south"]["social_value"],
            "south_empathy": all_action_breakdowns["move_south"]["empathy_trauma"],
            "south_sim_ob": all_action_breakdowns["move_south"]["sim_other_battery"],
            # MOVE EAST
            "east_pragmatic": all_action_breakdowns["move_east"]["pragmatic"],
            "east_threat": all_action_breakdowns["move_east"]["threat_penalty"],
            "east_battery": all_action_breakdowns["move_east"]["battery_penalty"],
            "east_wait": all_action_breakdowns["move_east"]["wait_penalty"],
            "east_commit": all_action_breakdowns["move_east"]["commitment_penalty"],
            "east_explore": all_action_breakdowns["move_east"]["exploration_bonus"],
            "east_social": all_action_breakdowns["move_east"]["social_value"],
            "east_empathy": all_action_breakdowns["move_east"]["empathy_trauma"],
            "east_sim_ob": all_action_breakdowns["move_east"]["sim_other_battery"],
            # MOVE WEST
            "west_pragmatic": all_action_breakdowns["move_west"]["pragmatic"],
            "west_threat": all_action_breakdowns["move_west"]["threat_penalty"],
            "west_battery": all_action_breakdowns["move_west"]["battery_penalty"],
            "west_wait": all_action_breakdowns["move_west"]["wait_penalty"],
            "west_commit": all_action_breakdowns["move_west"]["commitment_penalty"],
            "west_explore": all_action_breakdowns["move_west"]["exploration_bonus"],
            "west_social": all_action_breakdowns["move_west"]["social_value"],
            "west_empathy": all_action_breakdowns["move_west"]["empathy_trauma"],
            "west_sim_ob": all_action_breakdowns["move_west"]["sim_other_battery"],
            # WAIT
            "wait_pragmatic": all_action_breakdowns["wait"]["pragmatic"],
            "wait_threat": all_action_breakdowns["wait"]["threat_penalty"],
            "wait_battery": all_action_breakdowns["wait"]["battery_penalty"],
            "wait_wait": all_action_breakdowns["wait"]["wait_penalty"],
            "wait_commit": all_action_breakdowns["wait"]["commitment_penalty"],
            "wait_explore": all_action_breakdowns["wait"]["exploration_bonus"],
            "wait_social": all_action_breakdowns["wait"]["social_value"],
            "wait_empathy": all_action_breakdowns["wait"]["empathy_trauma"],
            "wait_sim_ob": all_action_breakdowns["wait"]["sim_other_battery"],
            # DROP ENERGY
            "drop_pragmatic": all_action_breakdowns["drop_energy"]["pragmatic"],
            "drop_threat": all_action_breakdowns["drop_energy"]["threat_penalty"],
            "drop_battery": all_action_breakdowns["drop_energy"]["battery_penalty"],
            "drop_wait": all_action_breakdowns["drop_energy"]["wait_penalty"],
            "drop_commit": all_action_breakdowns["drop_energy"]["commitment_penalty"],
            "drop_explore": all_action_breakdowns["drop_energy"]["exploration_bonus"],
            "drop_social": all_action_breakdowns["drop_energy"]["social_value"],
            "drop_empathy": all_action_breakdowns["drop_energy"]["empathy_trauma"],
            "drop_sim_ob": all_action_breakdowns["drop_energy"]["sim_other_battery"],

            # Component-preference analysis
            "best_pragmatic_action": best_pragmatic_action,
            "best_social_action": best_social_action,
            "best_empathy_action": best_empathy_action,

            # Social condition detail
            **social_detail,

            # --- FORWARD MODEL: PRE-ACTION STATE (raw 9-dim) ---
            "pre_dx_energy": round(raw_state_pre[0].item(), 4),
            "pre_dy_energy": round(raw_state_pre[1].item(), 4),
            "pre_dx_threat": round(raw_state_pre[2].item(), 4),
            "pre_dy_threat": round(raw_state_pre[3].item(), 4),
            "pre_dx_other": round(raw_state_pre[4].item(), 4),
            "pre_dy_other": round(raw_state_pre[5].item(), 4),
            "pre_battery": round(raw_state_pre[6].item(), 4),
            "pre_health": round(raw_state_pre[7].item(), 4),
            "pre_other_battery": round(raw_state_pre[8].item(), 4),

            # --- FORWARD MODEL: PREDICTED NEXT STATE (9-dim) ---
            "pred_dx_energy": round(predicted_next_state[0].item(), 4),
            "pred_dy_energy": round(predicted_next_state[1].item(), 4),
            "pred_dx_threat": round(predicted_next_state[2].item(), 4),
            "pred_dy_threat": round(predicted_next_state[3].item(), 4),
            "pred_dx_other": round(predicted_next_state[4].item(), 4),
            "pred_dy_other": round(predicted_next_state[5].item(), 4),
            "pred_battery": round(predicted_next_state[6].item(), 4),
            "pred_health": round(predicted_next_state[7].item(), 4),
            "pred_other_battery": round(predicted_next_state[8].item(), 4),

            # --- FORWARD MODEL: ACTUAL NEXT STATE (9-dim) ---
            "post_dx_energy": round(raw_state_post[0].item(), 4),
            "post_dy_energy": round(raw_state_post[1].item(), 4),
            "post_dx_threat": round(raw_state_post[2].item(), 4),
            "post_dy_threat": round(raw_state_post[3].item(), 4),
            "post_dx_other": round(raw_state_post[4].item(), 4),
            "post_dy_other": round(raw_state_post[5].item(), 4),
            "post_battery": round(raw_state_post[6].item(), 4),
            "post_health": round(raw_state_post[7].item(), 4),
            "post_other_battery": round(raw_state_post[8].item(), 4),

            # --- FORWARD MODEL: PREDICTION ACCURACY (per-dimension) ---
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
            "valence": round(valence, 4),

            # --- META-PLASTICITY ---
            "meta_lambda": round(dynamic_lambda, 6),
            "dynamic_lr": round(dynamic_lr, 6),
            "surprise_window_len": len(ica.meta_controller.error_window),
            "surprise_window_avg": round(sum(ica.meta_controller.error_window) / len(ica.meta_controller.error_window), 6) if ica.meta_controller.error_window else None,
            "rigidity_factor": round(min(max(1.0 / (1.0 + (sum(ica.meta_controller.error_window) / len(ica.meta_controller.error_window)) * 5.0), 0.1), 1.0), 6) if ica.meta_controller.error_window else None,

            # --- DYNAMIC WORLD (Entropy / Open-Ended Evolution) ---
            "entropy_cycle": world.entropy_counter,
            "entropy_event": world.entropy_counter > 0 and world.entropy_counter % 30 == 0,
            "physics_inverted": world.physics_inverted,
            "num_energy_tiles": len(world.energy_tiles),
            "energy_tiles": world.energy_tiles,
            "daytime": world.time_of_day < 50,

            # --- COGNITIVE MAP (hippocampal belief) ---
            "visited_cells": int(ica.cognitive_map.visited.sum().item()),
            "unvisited_cells": 49 - int(ica.cognitive_map.visited.sum().item()),
            "exploration_pct": round(int(ica.cognitive_map.visited.sum().item()) / 49 * 100, 1),
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
            "other_state": other.state,
            "social_collision": social_collision,
            "other_dist": abs(world.x - other.x) + abs(world.y - other.y),
            "other_dx": world.x - other.x,
            "other_dy": world.y - other.y,
            "other_health": other.health,

            # --- CORTICAL MODULE ACTIVATIONS ---
            **{f"mod_{k}": v for k, v in module_activations.items()},
            **module_outputs,

            # --- PIPELINE LATENCY ---
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

            # --- SLEEP / PLASTICITY EVENTS ---
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
        empathy_tag = ""
        if social_detail["social_drop_saves_other"]:
            empathy_tag = " [EMPATHY]"

        print(f"  C{entry['cycle']:2d}{inv_tag} | Bat={entry['battery_pct']:5.1f}% HP={entry['health_pct']:5.1f}% | "
              f"{entry['action']:12s}{depth_str}{src_tag}{social_tag}{empathy_tag} | PE={entry['prediction_error']:.3f} "
              f"Cur={entry['curiosity']:.3f} | NM={entry['neuromod']:.2f} "
              f"V={entry['visited_cells']:2d}/{49}"
              f" | O={entry['other_battery']:5.1f}%{ ' STARVING!' if entry['other_starving'] else ''}{' DIED!' if other_died else ''}"
              f"{' | DEAD' if dead else ''}")

        log_entries.append(entry)

        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        if dead:
            success_rate = ica.cycle_count / 60.0
            ica.genome.fitness = (ica.genome.fitness * 0.8) + (success_rate * 0.2)
            print(f"  >>> DIED at cycle {ica.cycle_count} ({world.steps_alive} steps | success_rate={success_rate:.2f})")
            ica.genome.mutate(success_rate)
            ica.genome.save()
            ica.check_and_trigger_neurogenesis()
            ica.synaptic_turnover(turnover_rate=0.10)
            ica.meta_controller.base_lambda = ica.genome.genes['ewc_base_lambda']
            ica.meta_controller.base_lr = ica.genome.genes['learning_rate']
            print(f"  [GENOME] lr={ica.genome.genes['learning_rate']:.6f} ewc_lambda={ica.genome.genes['ewc_base_lambda']:.2f} social={ica.genome.genes['social_value']:.2f} bat_pen={ica.genome.genes['battery_penalty']:.2f}")
            ica.cognitive_map.map.zero_()
            ica.cognitive_map.visited.zero_()
            ica.world.reset()
            ica.workspace.clear()
            for m in ica.modules.values():
                m.activation = 0.0
                m.last_output = ""
            ica.cycle_count = 0
            break

    return log_entries

def summarize_life(entries):
    if not entries:
        return {"error": "no_cycles"}
    pe_vals = [e["prediction_error"] for e in entries]
    cur_vals = [e["curiosity"] for e in entries]
    nm_vals = [e["neuromod"] for e in entries]
    deps = [e["planning_depth"] for e in entries if e["planning_depth"] is not None]
    repeats = [e["action_repeats"] for e in entries if e["action_repeats"] >= 2]
    altruism_cycle = [e["cycle"] for e in entries if e.get("altruism", False)]
    other_death_cycles = [e["cycle"] for e in entries if e.get("other_died", False)]
    north_count = sum(1 for e in entries if e.get("val_NORTH", -999) >= max(e.get("val_SOUTH",-999), e.get("val_EAST",-999), e.get("val_WEST",-999), e.get("val_WAIT",-999)))
    drop_best_count = sum(1 for e in entries if e.get("val_DROP", -999) >= max(e.get("val_NORTH",-999), e.get("val_SOUTH",-999), e.get("val_EAST",-999), e.get("val_WEST",-999), e.get("val_WAIT",-999)))
    social_conditions_met = sum(1 for e in entries if e.get("social_all_conditions_met", False))
    social_adjacent_starving = sum(1 for e in entries if e.get("social_adjacent_and_starving", False))

    # Analyze DROP ENERGY components
    drop_components = {k: [] for k in ["pragmatic","threat","battery","wait","commit","explore","social"]}
    for e in entries:
        for comp in drop_components:
            val = e.get(f"drop_{comp}", None)
            if val is not None:
                drop_components[comp].append(val)

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
        "avg_drop_pragmatic": round(sum(drop_components["pragmatic"]) / len(drop_components["pragmatic"]), 4) if drop_components["pragmatic"] else 0,
        "avg_drop_threat": round(sum(drop_components["threat"]) / len(drop_components["threat"]), 4) if drop_components["threat"] else 0,
        "avg_drop_battery": round(sum(drop_components["battery"]) / len(drop_components["battery"]), 4) if drop_components["battery"] else 0,
        "avg_drop_social": round(sum(drop_components["social"]) / len(drop_components["social"]), 4) if drop_components["social"] else 0,
    }

if __name__ == "__main__":
    if os.path.exists(LOG):
        os.remove(LOG)
    print(f"ICA Highly Detailed Diagnostic v3 - {RUNS} runs, {MAX_CYCLES} max cycles each")
    print(f"Model: {ica.model.__class__.__name__} ({sum(p.numel() for p in ica.model.parameters())} params)")
    start = time.time()
    results = run_diagnostic()
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print("FINAL AGGREGATED SUMMARY")
    print(f"{'='*60}")
    all_lifespans = [summarize_life(v)["cycles_lived"] for v in results.values()]
    all_pe = []
    all_cur = []
    all_nm = []
    all_visited = []
    all_val_spreads = []
    depth_counts = {}
    total_cycles = 0
    total_altruism = 0
    total_other_deaths = 0
    altruism_cycles_all = []
    other_death_cycles_all = []
    total_social_collisions = 0
    total_action_repeats = 0
    total_drop_best = 0
    total_social_cond = 0
    for rname, entries in results.items():
        s = summarize_life(entries)
        all_pe.extend(e["prediction_error"] for e in entries)
        all_cur.extend(e["curiosity"] for e in entries)
        all_nm.extend(e["neuromod"] for e in entries)
        all_visited.append(s["final_visited_cells"])
        total_altruism += s["altruism_acts"]
        total_other_deaths += s["other_deaths"]
        altruism_cycles_all.extend(s.get("altruism_at_cycles", []))
        other_death_cycles_all.extend(s.get("other_death_at_cycles", []))
        total_social_collisions += s["social_collisions"]
        total_action_repeats += s["action_repeat_events"]
        total_drop_best += s["drop_best_count"]
        total_social_cond += s["social_conditions_met"]
        for d, c in s["depth_use"].items():
            depth_counts[d] = depth_counts.get(d, 0) + c
        total_cycles += s["cycles_lived"]
        all_val_spreads.append(s["avg_val_spread"])
    print(f"  Lives: {RUNS}")
    print(f"  Total cycles: {total_cycles}")
    print(f"  Avg lifespan: {sum(all_lifespans)/len(all_lifespans):.1f} cycles")
    print(f"  Avg PE: {sum(all_pe)/len(all_pe):.6f}")
    print(f"  Avg Curiosity: {sum(all_cur)/len(all_cur):.6f}")
    print(f"  Avg Neuromod: {sum(all_nm)/len(all_nm):.6f}")
    print(f"  Depth distribution: {depth_counts}")
    print(f"  Total altruism acts: {total_altruism} at cycles {altruism_cycles_all}")
    print(f"  Total other-agent deaths: {total_other_deaths} at cycles {other_death_cycles_all}")
    print(f"  Times DROP was best at depth 0: {total_drop_best}")
    print(f"  Times social conditions were met: {total_social_cond}")
    print(f"  Avg visited cells: {sum(all_visited)/len(all_visited):.1f}/49")
    print(f"  Avg val spread: {sum(all_val_spreads)/len(all_val_spreads):.4f}")
    print(f"  Total social collisions: {total_social_collisions}")
    print(f"  Total action repeat events: {total_action_repeats}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"\nFull detailed log written to {LOG}")
    print(f"Approximately 200+ fields per entry")
