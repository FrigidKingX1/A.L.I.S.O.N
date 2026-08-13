"""
INTEGRATED CONSCIOUSNESS AGENT (ICA) -- Pure PyTorch
====================================================
A unified cognitive architecture combining:
- Global Workspace Theory (GWT)
- Integrated Information Theory (IIT)
- Higher-Order Theory (HOT)
- Active Inference (Free Energy Principle)
- EWC & Neuromodulated Plasticity
- Dream-based memory consolidation

Runs on CPU, no external dependencies beyond torch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Multiple threads (consciousness loop, affect loops, screen daemon, learners)
# run torch ops concurrently; the intra-op thread pool can livelock under
# concurrent parallel ops. Serialize torch's own threads to keep it stable.
torch.set_num_threads(1)
import random
import re
import math
import time
import os
import sys
import threading
import queue
import hashlib
import atexit
import json
import copy
import argparse
from collections import deque

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

try:
    import alison_sense
    HAS_SCREEN_SENSE = True
except ImportError:
    HAS_SCREEN_SENSE = False

try:
    import win32event, win32api, win32gui, win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

# ==================================================================
# 1. TOKENIZER
# ==================================================================
class CharTokenizer:
    def __init__(self):
        chars = [chr(i) for i in range(32, 127)]
        specials = ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]
        self.stoi = {s: i for i, s in enumerate(specials)}
        for i, c in enumerate(chars):
            self.stoi[c] = len(self.stoi)
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.pad_id = self.stoi["<PAD>"]
        self.unk_id = self.stoi["<UNK>"]

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_id]
        for ch in text:
            ids.append(self.stoi.get(ch, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            ch = self.itos.get(i, "<UNK>")
            if ch in ("<BOS>", "<EOS>", "<PAD>"):
                continue
            out.append(ch)
        return "".join(out).strip()

tokenizer = CharTokenizer()

# ==================================================================
# 2. LORA + TRANSFORMER
# ==================================================================
class LoRALinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, rank: int = 4, alpha: float = 4.0):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.lora_a = nn.Parameter(torch.randn(in_f, rank) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_f))
        self.alpha, self.rank = alpha, rank
        self.linear.requires_grad_(False)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_a @ self.lora_b) * (self.alpha / self.rank)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads, self.hdim = heads, dim // heads
        self.q = LoRALinear(dim, dim)
        self.k = LoRALinear(dim, dim)
        self.v = LoRALinear(dim, dim)
        self.out = LoRALinear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.heads, self.hdim).transpose(1, 2)
        k = self.k(x).view(B, T, self.heads, self.hdim).transpose(1, 2)
        v = self.v(x).view(B, T, self.heads, self.hdim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.hdim ** -0.5)
        att = att + torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach().cpu()
        return self.out((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        h = dim * 4
        self.gate = LoRALinear(dim, h)
        self.down = LoRALinear(h, dim)

    def forward(self, x):
        return self.down(F.gelu(self.gate(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, heads)
        self.ff = FeedForward(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class AgentLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, heads: int = 4,
                 layers: int = 4, max_seq: int = 512, grid_size: int = 7):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.x_emb = nn.Embedding(grid_size, dim)
        self.y_emb = nn.Embedding(grid_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight
        self.max_seq = max_seq
        pe = torch.zeros(max_seq, dim)
        pos = torch.arange(0, max_seq).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_embed', pe.unsqueeze(0))

    def forward(self, input_ids, labels=None, spatial_coords=None, grounded_state=None):
        B, T = input_ids.shape
        x = self.token_embed(input_ids) + self.pos_embed[:, :T, :]
        if spatial_coords is not None:
            sx = self.x_emb(torch.tensor(spatial_coords[0], device=input_ids.device))
            sy = self.y_emb(torch.tensor(spatial_coords[1], device=input_ids.device))
            x = x + (sx + sy).unsqueeze(0).unsqueeze(0)
        if grounded_state is not None:
            sensory_token = grounded_state.unsqueeze(0).unsqueeze(0)
            x = torch.cat([sensory_token, x], dim=1)
            T += 1
            if labels is not None:
                labels = torch.cat([torch.full((1, 1), -100, dtype=torch.long, device=input_ids.device), labels], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits, loss

    def forward_return_hidden(self, input_ids, spatial_coords=None, grounded_state=None):
        """Forward pass returning (logits, hidden_states) where hidden_states is the
        128-dim tensor after ln_f but before lm_head."""
        B, T = input_ids.shape
        x = self.token_embed(input_ids) + self.pos_embed[:, :T, :]
        if spatial_coords is not None:
            sx = self.x_emb(torch.tensor(spatial_coords[0], device=input_ids.device))
            sy = self.y_emb(torch.tensor(spatial_coords[1], device=input_ids.device))
            x = x + (sx + sy).unsqueeze(0).unsqueeze(0)
        if grounded_state is not None:
            sensory_token = grounded_state.unsqueeze(0).unsqueeze(0)
            x = torch.cat([sensory_token, x], dim=1)
            T += 1
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, x

    def forward_continuous(self, x, labels=None):
        """Forward pass with pre-embedded continuous states (bypasses token_embed).
        x: [B, T, d_model] — already embedded continuous states.
        Returns (logits, hidden_states)."""
        B, T = x.shape[:2]
        x = x + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits, x

    @torch.no_grad()
    def generate(self, input_ids, max_new=40, temp=0.6, spatial_coords=None, grounded_state=None):
        device = next(self.parameters()).device
        curr = input_ids.clone().to(device)
        for _ in range(max_new):
            ctx = curr[:, -self.max_seq:]
            logits, _ = self.forward(ctx, spatial_coords=spatial_coords, grounded_state=grounded_state)
            logits = logits[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            curr = torch.cat([curr, nxt], dim=1)
            if nxt.item() == tokenizer.eos_id:
                break
        return curr

# ==================================================================
# 3. BUILD MODEL
# ==================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = AgentLM(vocab_size=tokenizer.vocab_size, dim=128, heads=4, layers=4, max_seq=512).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for name, p in model.named_parameters():
    if 'lora_' not in name:
        p.requires_grad_(False)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {total:,} params ({trainable:,} trainable LoRA)")

# ==================================================================
# 4. GENERATION HELPERS
# ==================================================================
def generate_text(prompt, max_tokens=30, temp=0.4, spatial=None, grounded_state=None):
    with model_lock:
        ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        out = model.generate(ids, max_new=max_tokens, temp=temp, spatial_coords=spatial, grounded_state=grounded_state)
        full = tokenizer.decode(out[0].tolist())
    return full[len(tokenizer.decode(ids[0].tolist())):].strip()


# ==================================================================
# 5. PERCEPTUAL ENCODER (Symbol Grounding)
# ==================================================================
class PerceptualEncoder(nn.Module):
    """Converts raw world geometry into a 128-dim continuous 'feeling'."""
    def __init__(self, d_model=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 32),
            nn.GELU(),
            nn.Linear(32, d_model)
        )

    def forward(self, world):
        state = torch.tensor([
            (world.energy[0] - world.x) / 6.0,
            (world.energy[1] - world.y) / 6.0,
            (world.threat[0] - world.x) / 6.0,
            (world.threat[1] - world.y) / 6.0,
            (other_agent.x - world.x) / 6.0,
            (other_agent.y - world.y) / 6.0,
            world.battery / 100.0,
            world.health / 100.0,
            other_agent.battery / 100.0
        ], dtype=torch.float32).to(device)
        return self.fc(state)

perceptual_encoder = PerceptualEncoder(d_model=128).to(device)


# ==================================================================
# 5b. SPATIAL COGNITIVE MAP (Hippocampal Place/Grid Cells)
# ==================================================================
class SpatialCognitiveMap:
    """Maintains an absolute 7x7 belief state of the world."""
    def __init__(self, grid_size=7):
        self.grid_size = grid_size
        self.map = torch.zeros((3, grid_size, grid_size), device=device)
        self.visited = torch.zeros((grid_size, grid_size), device=device)

    def update_from_observation(self, world, other):
        ex, ey = world.energy
        tx, ty = world.threat
        self.map[0] *= 0.9
        if abs(world.x - ex) + abs(world.y - ey) <= 2:
            self.map[0, ey, ex] = 1.0
        self.map[1] *= 0.9
        if abs(world.x - tx) + abs(world.y - ty) <= 2:
            self.map[1, ty, tx] = 1.0
        self.map[2] *= 0.9
        self.map[2, other.y, other.x] = 1.0
        self.visited[world.y, world.x] = 1.0

    def get_map_vector(self):
        full_map = torch.cat([self.map, self.visited.unsqueeze(0)], dim=0)
        return full_map.view(-1)

    def update_from_step(self, old_x, old_y, new_x, new_y):
        pass


class PerceptualEncoderV2(nn.Module):
    """Symbol grounding with cognitive map injection."""
    def __init__(self, d_model=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(205, 64),
            nn.GELU(),
            nn.Linear(64, d_model)
        )

    def forward(self, world, cog_map_vec):
        raw_state = torch.tensor([
            (world.energy[0] - world.x) / 6.0, (world.energy[1] - world.y) / 6.0,
            (world.threat[0] - world.x) / 6.0, (world.threat[1] - world.y) / 6.0,
            (other_agent.x - world.x) / 6.0, (other_agent.y - world.y) / 6.0,
            world.battery / 100.0, world.health / 100.0,
            other_agent.battery / 100.0
        ], dtype=torch.float32).to(device)
        combined = torch.cat([raw_state, cog_map_vec])
        return self.fc(combined).squeeze(0)

cognitive_map = SpatialCognitiveMap(grid_size=7)
perceptual_encoder_v2 = PerceptualEncoderV2(d_model=128).to(device)

action_to_idx = {"MOVE NORTH": 0, "MOVE SOUTH": 1, "MOVE EAST": 2, "MOVE WEST": 3, "WAIT": 4, "DROP ENERGY": 5, "BUILD WALL": 6}


def get_raw_state(world):
    return torch.tensor([
        (world.energy[0] - world.x) / 6.0, (world.energy[1] - world.y) / 6.0,
        (world.threat[0] - world.x) / 6.0, (world.threat[1] - world.y) / 6.0,
        (other_agent.x - world.x) / 6.0, (other_agent.y - world.y) / 6.0,
        world.battery / 100.0, world.health / 100.0,
        other_agent.battery / 100.0
    ], dtype=torch.float32).to(device)


class SensoryForwardModel(nn.Module):
    """Predicts the next continuous sensory state based on current state and action."""
    def __init__(self, state_dim=9, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.GELU(),
            nn.Linear(32, state_dim)
        )
        self.opt = torch.optim.Adam(self.parameters(), lr=0.005)

    def predict_next_state(self, current_raw_state, action_idx):
        action_onehot = F.one_hot(torch.tensor(action_idx, device=device), num_classes=7).float()
        inp = torch.cat([current_raw_state, action_onehot])
        return self.net(inp)

    def calculate_latent_fe(self, current_raw_state, action_idx, actual_next_raw_state):
        pred_next = self.predict_next_state(current_raw_state, action_idx)
        loss = F.mse_loss(pred_next, actual_next_raw_state)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


sensory_forward_model = SensoryForwardModel().to(device)


def evaluate_continuous_state_v4(pred_raw_state, action_taken, current_battery, world, consecutive_count):
    """Evaluates utility with beaconing and diversification penalty."""
    dx_e, dy_e = pred_raw_state[0].item(), pred_raw_state[1].item()
    dx_t, dy_t = pred_raw_state[2].item(), pred_raw_state[3].item()
    battery = pred_raw_state[6].item()

    dist_to_energy = abs(dx_e) + abs(dy_e)
    pragmatic_value = 1.0 - (dist_to_energy / 12.0)

    dist_to_threat = abs(dx_t) + abs(dy_t)
    threat_penalty = -1.5 * (1.0 - (dist_to_threat / 12.0))

    starvation_risk = max(0.0, 1.0 - battery)
    battery_penalty = -1.0 * (starvation_risk ** 2)

    wait_penalty = -2.0 if (action_taken == "WAIT" and current_battery < 40) else 0.0

    beacon_bonus = 0.0
    if current_battery < 50:
        real_dist_now = abs(world.energy[0] - world.x) + abs(world.energy[1] - world.y)
        sim_x, sim_y = world.x, world.y
        if "NORTH" in action_taken and sim_y > 0: sim_y -= 1
        elif "SOUTH" in action_taken and sim_y < 6: sim_y += 1
        elif "EAST" in action_taken and sim_x < 6: sim_x += 1
        elif "WEST" in action_taken and sim_x > 0: sim_x -= 1
        real_dist_next = abs(world.energy[0] - sim_x) + abs(world.energy[1] - sim_y)
        if real_dist_next < real_dist_now:
            beacon_bonus = 1.5

    commitment_penalty = 0.0
    if consecutive_count >= 2:
        commitment_penalty = -0.8 * (consecutive_count - 1)

    return pragmatic_value + threat_penalty + battery_penalty + wait_penalty + beacon_bonus + commitment_penalty


def evaluate_continuous_state_v5(pred_raw_state, action_taken, current_battery, world, consecutive_count, cog_map):
    """Adds spatial mapping and proactive exploration drive."""
    dx_e, dy_e = pred_raw_state[0].item(), pred_raw_state[1].item()
    dx_t, dy_t = pred_raw_state[2].item(), pred_raw_state[3].item()
    battery = pred_raw_state[6].item()

    dist_to_energy = abs(dx_e) + abs(dy_e)
    pragmatic_value = 1.0 - (dist_to_energy / 12.0)

    dist_to_threat = abs(dx_t) + abs(dy_t)
    threat_penalty = -1.5 * (1.0 - (dist_to_threat / 12.0))

    starvation_risk = max(0.0, 1.0 - battery)
    battery_penalty = -1.0 * (starvation_risk ** 2)

    wait_penalty = -2.0 if (action_taken == "WAIT" and current_battery < 40) else 0.0

    beacon_bonus = 0.0
    if current_battery < 50:
        real_dist_now = abs(world.energy[0] - world.x) + abs(world.energy[1] - world.y)
        sim_x, sim_y = world.x, world.y
        if "NORTH" in action_taken and sim_y > 0: sim_y -= 1
        elif "SOUTH" in action_taken and sim_y < 6: sim_y += 1
        elif "EAST" in action_taken and sim_x < 6: sim_x += 1
        elif "WEST" in action_taken and sim_x > 0: sim_x -= 1
        real_dist_next = abs(world.energy[0] - sim_x) + abs(world.energy[1] - sim_y)
        if real_dist_next < real_dist_now:
            beacon_bonus = 1.5

    commitment_penalty = -0.8 * (consecutive_count - 1) if consecutive_count >= 2 else 0.0

    exploration_bonus = 0.0
    if "NORTH" in action_taken and world.y > 0:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y - 1, world.x].item())
    elif "SOUTH" in action_taken and world.y < 6:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y + 1, world.x].item())
    elif "EAST" in action_taken and world.x < 6:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y, world.x + 1].item())
    elif "WEST" in action_taken and world.x > 0:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y, world.x - 1].item())

    return pragmatic_value + threat_penalty + battery_penalty + wait_penalty + beacon_bonus + commitment_penalty + exploration_bonus


def evaluate_continuous_state_v6(pred_raw_state, action_taken, current_battery, world, consecutive_count, cog_map, other_agent):
    """V5 + emergent social value: DROP ENERGY competes mathematically against MOVE/WAIT."""
    dx_e, dy_e = pred_raw_state[0].item(), pred_raw_state[1].item()
    dx_t, dy_t = pred_raw_state[2].item(), pred_raw_state[3].item()
    battery = pred_raw_state[6].item()

    dist_to_energy = abs(dx_e) + abs(dy_e)
    pragmatic_value = 1.0 - (dist_to_energy / 12.0)
    dist_to_threat = abs(dx_t) + abs(dy_t)
    threat_penalty = -1.5 * (1.0 - (dist_to_threat / 12.0))
    starvation_risk = max(0.0, 1.0 - battery)
    battery_penalty = -1.0 * (starvation_risk ** 2)
    wait_penalty = -2.0 if (action_taken == "WAIT" and current_battery < 40) else 0.0

    commitment_penalty = -0.8 * (consecutive_count - 1) if consecutive_count >= 2 else 0.0

    exploration_bonus = 0.0
    if "NORTH" in action_taken and world.y > 0:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y - 1, world.x].item())
    elif "SOUTH" in action_taken and world.y < 6:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y + 1, world.x].item())
    elif "EAST" in action_taken and world.x < 6:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y, world.x + 1].item())
    elif "WEST" in action_taken and world.x > 0:
        exploration_bonus = 0.3 * (1.0 - cog_map.visited[world.y, world.x - 1].item())

    social_value = 0.0
    if action_taken == "DROP ENERGY":
        dist_to_other = abs(world.x - other_agent.x) + abs(world.y - other_agent.y)
        if other_agent.is_starving and dist_to_other <= 1 and world.battery > 40:
            social_value = 2.0
        else:
            social_value = -2.0

    return pragmatic_value + threat_penalty + battery_penalty + wait_penalty + commitment_penalty + exploration_bonus + social_value


def evaluate_continuous_state_v7(pred_raw_state, action_taken, current_battery, world, consecutive_count, cog_map, sim_other_battery, sim_dist_to_other, other_is_hostile=False):
    """V7: evolved moral values from genome with self-preservation imperative + hostility threat."""
    # -- SELF-PRESERVATION IMPERATIVE --
    if action_taken == "BUILD WALL" and current_battery <= 20:
        return -10.0
    if action_taken == "DROP ENERGY" and current_battery - 20 < 10:
        if sim_other_battery > 0:
            return -10.0
    if action_taken == "WAIT" and current_battery < 10:
        return -10.0

    dx_e, dy_e = pred_raw_state[0].item(), pred_raw_state[1].item()
    dx_t, dy_t = pred_raw_state[2].item(), pred_raw_state[3].item()
    battery = pred_raw_state[6].item()

    dist_to_energy = abs(dx_e) + abs(dy_e)
    pragmatic_value = 1.0 - (dist_to_energy / 12.0)

    # AFFECT BIASES from LimbicSystem (Phase 24)
    biases = limbic_system.affect_biases()

    dist_to_threat = abs(dx_t) + abs(dy_t)
    threat_penalty = -1.5 * biases['threat_sensitivity'] * (1.0 - (dist_to_threat / 12.0))
    starvation_risk = max(0.0, 1.0 - battery)
    battery_penalty = genome.genes['battery_penalty'] * (starvation_risk ** 2)
    wait_penalty = -2.0 if (action_taken == "WAIT" and current_battery < 40) else 0.0

    commitment_penalty = -0.8 * (consecutive_count - 1) if consecutive_count >= 2 else 0.0
    exploration_bonus = 0.0
    max_idx = world.grid_size - 1
    if "NORTH" in action_taken and world.y > 0:
        exploration_bonus = biases['exploration_bonus'] * (1.0 - cog_map.visited[world.y - 1, world.x].item())
    elif "SOUTH" in action_taken and world.y < max_idx:
        exploration_bonus = biases['exploration_bonus'] * (1.0 - cog_map.visited[world.y + 1, world.x].item())
    elif "EAST" in action_taken and world.x < max_idx:
        exploration_bonus = biases['exploration_bonus'] * (1.0 - cog_map.visited[world.y, world.x + 1].item())
    elif "WEST" in action_taken and world.x > 0:
        exploration_bonus = biases['exploration_bonus'] * (1.0 - cog_map.visited[world.y, world.x - 1].item())

    # DYNAMIC SOCIAL VALUES modulated by hostility + affect
    social_value = 0.0
    if action_taken == "DROP ENERGY":
        if world.battery > 40 and sim_dist_to_other <= 1:
            if other_is_hostile:
                social_value = 4.0
            else:
                safety_margin = (world.battery - 20) / 80.0
                social_value = genome.genes['social_value'] * safety_margin * biases['social_boost']
        else:
            social_value = -2.0

    empathy_trauma = 0.0
    if sim_other_battery <= 0:
        empathy_trauma = -5.0 * genome.genes['social_value'] * biases['social_boost']

    # Wall bonus scaled by battery level
    wall_bonus = 0.0
    if action_taken == "BUILD WALL":
        if current_battery > 40:
            dist_to_threat_now = abs(world.threat[0] - world.x) + abs(world.threat[1] - world.y)
            if (world.x, world.y) in world.walls:
                wall_bonus = -2.0
            elif dist_to_threat_now <= 2:
                wall_bonus = 0.8
        else:
            wall_bonus = -2.0

    # HOSTILITY THREAT PENALTY: hostile other close and NOT being fed
    hostility_threat = 0.0
    if other_is_hostile and sim_dist_to_other <= 1 and action_taken != "DROP ENERGY":
        hostility_threat = -3.0

    return pragmatic_value + threat_penalty + battery_penalty + wait_penalty + commitment_penalty + exploration_bonus + social_value + empathy_trauma + wall_bonus + hostility_threat


def adaptive_deep_planning_v4(world, sensory_forward_model, current_raw_state, last_real_action, cog_map, other_agent):
    """Greedy planning with DROP ENERGY in the mathematical tree search (no heuristic)."""
    if world.battery < 30:
        depth = 5
    elif world.battery < 60:
        depth = 4
    else:
        depth = 2
    actions = ["MOVE NORTH", "MOVE SOUTH", "MOVE EAST", "MOVE WEST", "WAIT", "DROP ENERGY"]
    best_action = "WAIT"
    best_value = -999.0
    current_state = current_raw_state
    with torch.no_grad():
        for step in range(depth):
            step_best_action = "WAIT"
            step_best_value = -999.0
            next_state_for_best = None
            for action in actions:
                if action == "DROP ENERGY":
                    pred_state = current_state.clone()
                    pred_state[6] = max(0.0, pred_state[6].item() - 0.2)
                else:
                    pred_state = sensory_forward_model.predict_next_state(current_state, action_to_idx[action]).detach()
                consecutive_count = 1 if (step == 0 and action == last_real_action) else 0
                val = evaluate_continuous_state_v6(pred_state, action, world.battery, world, consecutive_count, cog_map, other_agent)
                if val > step_best_value:
                    step_best_value = val
                    step_best_action = action
                    next_state_for_best = pred_state
            if step == 0:
                best_action = step_best_action
                best_value = step_best_value
            current_state = next_state_for_best
    return best_action, best_value


def adaptive_deep_planning_v5(world, sensory_forward_model, current_raw_state, last_real_action, cog_map, other_agent):
    """Joint Latent Simulation: simulates Other Agent movement + battery across all depth steps."""
    base_depth = genome.genes['planning_depth']
    if world.battery < 30:
        depth = base_depth + 2
    elif world.battery < 60:
        depth = base_depth + 1
    else:
        depth = base_depth
    depth = max(1, depth)
    actions = ["MOVE NORTH", "MOVE SOUTH", "MOVE EAST", "MOVE WEST", "WAIT", "DROP ENERGY", "BUILD WALL"]
    best_action = "WAIT"
    best_value = -999.0
    current_state = current_raw_state
    sim_other_bat = other_agent.battery
    sim_other_x, sim_other_y = other_agent.x, other_agent.y
    with torch.no_grad():
        for step in range(depth):
            step_best_action = "WAIT"
            step_best_value = -999.0
            next_state_for_best = None
            sim_other_bat_for_best = sim_other_bat
            sim_other_pos_for_best = (sim_other_x, sim_other_y)
            for action in actions:
                # Simulate other agent moving toward energy
                sox, soy = sim_other_x, sim_other_y
                if soy < world.energy[1]: soy += 1
                elif soy > world.energy[1]: soy -= 1
                elif sox < world.energy[0]: sox += 1
                elif sox > world.energy[0]: sox -= 1
                dist_to_sim_other = abs(world.x - sim_other_x) + abs(world.y - sim_other_y)
                if action == "DROP ENERGY":
                    pred_state = current_state.clone()
                    pred_state[6] = max(0.0, pred_state[6].item() - 0.2)
                    sim_ob_next = min(100.0, sim_other_bat + 30.0) if dist_to_sim_other <= 1 else sim_other_bat - 3.0
                elif action == "BUILD WALL":
                    pred_state = current_state.clone()
                    pred_state[6] = max(0.0, pred_state[6].item() - 0.05)
                    sim_ob_next = sim_other_bat - 3.0
                else:
                    pred_state = sensory_forward_model.predict_next_state(current_state, action_to_idx[action]).detach()
                    sim_ob_next = sim_other_bat - 3.0
                sim_dist_other = abs(world.x - sox) + abs(world.y - soy)
                consecutive_count = 1 if (step == 0 and action == last_real_action) else 0
                val = evaluate_continuous_state_v7(pred_state, action, world.battery, world, consecutive_count, cog_map, sim_ob_next, sim_dist_other, other_is_hostile=other_agent.is_hostile)
                if val > step_best_value:
                    step_best_value = val
                    step_best_action = action
                    next_state_for_best = pred_state
                    sim_other_bat_for_best = sim_ob_next
                    sim_other_pos_for_best = (sox, soy)
            if step == 0:
                best_action = step_best_action
                best_value = step_best_value
            current_state = next_state_for_best
            sim_other_bat = sim_other_bat_for_best
            sim_other_x, sim_other_y = sim_other_pos_for_best
    return best_action, best_value


def execute_drop_energy(world, other_agent):
    """Transfer energy to starving other agent. Returns social reward."""
    if world.battery > 30:
        world.battery -= 20
        other_agent.battery = min(100, other_agent.battery + 30)
        other_agent.is_starving = False
        return 1.0
    return 0.0


def adaptive_deep_planning_v3(world, sensory_forward_model, current_raw_state, last_real_action):
    """Greedy planning with sensory beaconing and diversification penalty."""
    if world.battery < 30:
        depth = 5
    elif world.battery < 60:
        depth = 4
    else:
        depth = 2
    actions = ["MOVE NORTH", "MOVE SOUTH", "MOVE EAST", "MOVE WEST", "WAIT"]
    best_action = "WAIT"
    best_value = -999.0
    current_state = current_raw_state
    with torch.no_grad():
        for step in range(depth):
            step_best_action = "WAIT"
            step_best_value = -999.0
            next_state_for_best = None
            for action in actions:
                pred_state = sensory_forward_model.predict_next_state(current_state, action_to_idx[action]).detach()
                consecutive_count = 1 if (step == 0 and action == last_real_action) else 0
                val = evaluate_continuous_state_v4(pred_state, action, world.battery, world, consecutive_count)
                if val > step_best_value:
                    step_best_value = val
                    step_best_action = action
                    next_state_for_best = pred_state
            if step == 0:
                best_action = step_best_action
                best_value = step_best_value
            current_state = next_state_for_best
    return best_action, best_value


class CuriosityModule:
    def __init__(self):
        self.predicted_state = None

    def set_prediction(self, predicted_state):
        self.predicted_state = predicted_state.detach()

    def calculate_curiosity(self, actual_raw_state):
        if self.predicted_state is None:
            return 0.0
        novelty = F.mse_loss(self.predicted_state, actual_raw_state).item()
        return min(1.0, novelty * 2.0)


curiosity_module = CuriosityModule()


def visualize_attention(model, tokenizer, prompt_text, layer=0, head=0, top_k=5):
    """Prints the top-k attended tokens from the last token's perspective."""
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt_text)]).to(device)
    with torch.no_grad():
        model.forward(ids)

    attn = model.blocks[layer].attn.last_attn
    att_head = attn[0, head]
    last_idx = att_head.shape[0] - 1
    att_from_last = att_head[last_idx]

    tokens = list(tokenizer.decode(ids[0].tolist()))
    ids_list = ids[0].tolist()
    # Map each token ID to its character, showing special tokens
    token_labels = []
    for tid in ids_list:
        if tid == tokenizer.bos_id:
            token_labels.append("[BOS]")
        elif tid == tokenizer.eos_id:
            token_labels.append("[EOS]")
        else:
            ch = tokenizer.itos.get(tid, "<?>")
            token_labels.append(repr(ch) if ch == " " else ch)

    vals, idxs = torch.topk(att_from_last, min(top_k, len(token_labels)))
    print(f"\n  [ATTN L{layer}H{head}] Top {top_k} tokens attended from last position:")
    for i in range(len(idxs)):
        ti = idxs[i].item()
        print(f"    {token_labels[ti]:>8}  weight={vals[i].item():.3f}")


# ==================================================================
# 5. RIGOROUS EWC (Fisher Information Matrix)
# ==================================================================
fisher_matrix = {}
ewc_optimal_weights = {}

def update_ewc_fisher(experiences):
    """Calculates and accumulates the Fisher Information Matrix after learning."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    for exp in experiences[-3:]:
        input_ids, labels = encode_pair(exp['prompt'], exp['response'])
        model.zero_grad()
        _, loss = model(input_ids, labels)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data.pow(2) / len(experiences)

    for n in fisher:
        if n in fisher_matrix:
            fisher_matrix[n] += fisher[n]
        else:
            fisher_matrix[n] = fisher[n]

    for n, p in model.named_parameters():
        if p.requires_grad:
            ewc_optimal_weights[n] = p.data.clone()

def apply_ewc_penalty(loss, ewc_lambda=0.01):
    """Adds an L2 penalty that resists changes to Fisher-important weights."""
    if not fisher_matrix:
        return loss
    penalty = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and n in fisher_matrix:
            drift = p - ewc_optimal_weights[n]
            penalty += (fisher_matrix[n] * drift.pow(2)).sum()
    return loss + (ewc_lambda * penalty)


# ==================================================================
# 6. WAKE-SLEEP CONSOLIDATION (Hippocampal Replay)
# ==================================================================
hippocampal_buffer = []
is_sleeping = False

# ==================================================================
# PHASE 24: THE LIMBIC SYSTEM (Continuous Affective Core)
# ==================================================================

class LimbicSystem(nn.Module):
    """Continuous State-Space Limbic System. Takes 6-dim PC state vectors,
    encodes them to 128-dim, passes through the 842K core, and maintains
    a 128-dim affect_vector via EMA on PRE-LN hidden states (preserves variance).
    Owns its own EWC state (fisher_matrix, optimal_weights)."""

    def __init__(self, core_net, state_dim=6, d_model=128,
                 module_fisher=None, module_optimal=None,
                 mood_classifier=None):
        super().__init__()
        self.core_net = core_net
        self.d_model = d_model
        self.state_encoder = nn.Linear(state_dim, d_model)
        self.register_buffer('affect_vector', torch.zeros(1, 1, d_model))
        self.optimizer = torch.optim.AdamW(
            [p for n, p in core_net.named_parameters() if p.requires_grad] +
            list(self.state_encoder.parameters()), lr=1e-4, weight_decay=1e-5)
        # Link to module-level Fisher dicts (Python reference semantics)
        # When module_fisher is passed, self.fisher_matrix IS the module-level dict
        self.fisher_matrix = module_fisher if module_fisher is not None else {}
        self.optimal_weights = module_optimal if module_optimal is not None else {}
        self._ewc_loss_window = deque(maxlen=100)
        self.mood_classifier = mood_classifier if mood_classifier is not None else MoodClassifier()

    def _forward_blocks(self, pc_state_vector):
        """Manual forward pass through transformer blocks.
        Supports both single [dim] and batched [B, dim] inputs."""
        embedded = self.state_encoder(pc_state_vector)
        if embedded.dim() == 1:
            embedded = embedded.unsqueeze(0)
        embedded = embedded.unsqueeze(1)
        x = embedded + self.core_net.pos_embed[:, :1, :]
        for block in self.core_net.blocks:
            x = block(x)
        return x  # pre-LN hidden state

    def compute_affect(self, pc_state_vector):
        """Pure forward pass, no EMA. Returns [1,1,128] via pre-LN hidden."""
        with model_lock:
            self.core_net.eval()
            with torch.no_grad():
                x = self._forward_blocks(pc_state_vector)
                raw_hidden = x[:, -1:, :128].clone()
        return raw_hidden

    def update_affect(self, pc_state_vector):
        """Extracts PRE-LN hidden states to preserve affective variance.
        EMA blend: 0.8 * old + 0.2 * new."""
        new_affect = self.compute_affect(pc_state_vector)
        self.affect_vector = (0.8 * self.affect_vector) + (0.2 * new_affect)
        # Active Inference: feed prediction error -> dynamic precision (gamma)
        if active_inference is not None:
            cur = self.affect_vector.detach().cpu().float().flatten().unsqueeze(0)
            if active_inference._prev is not None:
                active_inference.update_precision(active_inference._prev, cur)
            active_inference._prev = cur

    def compute_fisher(self, critical_states, n_noisy=16):
        """Compute True Fisher Information Matrix from critical PC states.
        For each critical state, generates n_noisy noisy variants and computes
        MSE against the FIXED one-hot target (not the model's own prediction),
        so gradients cannot vanish even when anchors are fully fit. Adds a
        1e-6 floor to guarantee non-zero curvature for the EWC penalty."""
        with model_lock:
            self.fisher_matrix.clear()
            self.optimal_weights.clear()
            self.core_net.train()
            for n, p in self.core_net.named_parameters():
                if p.requires_grad:
                    self.fisher_matrix[n] = torch.zeros_like(p.data)
                    self.optimal_weights[n] = p.data.clone().detach()
            n_states = len(critical_states)
            total_samples = 0
            for i, state in enumerate(critical_states):
                state = state.to(critical_states.device)
                target = torch.eye(self.d_model, self.d_model, device=state.device)[i].unsqueeze(0)
                for _ in range(n_noisy):
                    noisy_state = torch.clamp(state + torch.randn_like(state) * 0.15, 0.0, 1.0)
                    self.optimizer.zero_grad()
                    x = self._forward_blocks(noisy_state)
                    hidden = x[:, -1, :self.d_model]
                    loss = F.mse_loss(hidden, target)
                    loss.backward()
                    for n, p in self.core_net.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            self.fisher_matrix[n] += p.grad.data ** 2
                    total_samples += 1
            for n in self.fisher_matrix:
                self.fisher_matrix[n] = self.fisher_matrix[n] / total_samples + 1e-6
            self.core_net.eval()

    def learn_continuously(self, pc_state_vector, target_affect, ewc_lambda=0.01):
        """EWC-protected backprop. Learns to produce target_affect from pc_state_vector."""
        with model_lock:
            self.core_net.train()
            self.optimizer.zero_grad()
            x = self._forward_blocks(pc_state_vector)
            hidden = x[:, -1, :128]
            loss = F.mse_loss(hidden, target_affect.squeeze(0))
            ewc_penalty = 0.0
            if self.fisher_matrix:
                for n, p in self.core_net.named_parameters():
                    if n in self.fisher_matrix and p.requires_grad:
                        ewc_penalty = ewc_penalty + (self.fisher_matrix[n] * (p - self.optimal_weights[n]) ** 2).sum()
            total_loss = loss + (ewc_lambda * ewc_penalty)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.core_net.parameters(), 1.0)
            self.optimizer.step()
            self.core_net.eval()
            loss_val = loss.item()
            self._ewc_loss_window.append(loss_val)
        return loss_val

    def _get_v6(self):
        """Down-project the 128-dim affect to 6 dims for display/evaluator."""
        return self.mood_classifier(self.affect_vector.squeeze(0).squeeze(0))

    def train_down_proj(self, num_samples=200, lr=1e-2, epochs=500):
        """Deprecated — use MoodClassifier + train_mood_classifier_v3 instead."""
        print("  [WARN] train_down_proj is deprecated — MoodClassifier handles mood projection")

    def get_affect_prompt(self):
        """Translate affect into a mood sentence."""
        v6 = self._get_v6()
        parts = []
        labels = ["intense digital hunger and urgency", "pain from recent threat collisions",
                  "deep mental fatigue", "high curiosity and energy",
                  "anxiety from prolonged silence", "generosity and social inclination"]
        for i, label in enumerate(labels):
            if v6[i].item() > 0.3:
                parts.append(label)
        return "Your internal state: " + ". ".join(parts) + "." if parts else "Your internal state: stability and calm observation."

    def get_mood_label(self):
        """Dominant mood label."""
        v6 = self._get_v6()
        return ["HUNGRY", "PAIN", "FATIGUED", "CURIOUS", "ANXIOUS", "ALTRUISTIC"][torch.argmax(v6).item()]

    def get_persona_dict(self):
        v6 = self._get_v6()
        idx = torch.argmax(v6).item()
        moods = ["desperate and hungry", "wounded and alert", "fatigued and drowsy",
                 "curious and energized", "anxious and vigilant", "kind and generous"]
        energies = ["critical", "low", "depleted", "high", "moderate", "optimal"]
        shorts = ["hungry", "pained", "fatigued", "curious", "anxious", "altruistic"]
        return {"traits": f"curious, analytical, concise, {shorts[idx]}", "mood": moods[idx],
                "energy_level": energies[idx], "anxiety": round(v6[4].item(), 2),
                "pain": round(v6[1].item(), 2), "hunger": round(v6[0].item(), 2),
                "altruism": round(v6[5].item(), 2)}

    def affect_biases(self):
        v6 = self._get_v6()
        return {'exploration_bonus': max(0.0, v6[3].item() - 0.1),
                'threat_sensitivity': 1.0 + v6[1].item(),
                'social_boost': v6[5].item() * 0.5}

    def _get_alison_drives(self):
        """Map the canonical 6-dim limbic affect (v6) to the 6 A.L.I.S.O.N.
        homeostatic drives: PLEASURE, AROUSAL, ANXIETY, CURIOSITY,
        GOAL_URGENCY, SATIATION. Returns a list of 6 floats in [0, 1]."""
        v6 = self._get_v6()
        h, p, f, c, a, s = (v6[0].item(), v6[1].item(), v6[2].item(),
                            v6[3].item(), v6[4].item(), v6[5].item())

        def _c(x):
            return max(0.0, min(1.0, float(x)))

        pleasure = _c(0.5 * (s + (1.0 - p)))
        arousal = _c(0.6 * c + 0.4 * (1.0 - f))
        anxiety = _c(a)
        curiosity = _c(c)
        goal_urgency = _c(h)
        satiation = _c(0.5 * s + 0.5 * (1.0 - h))
        return [pleasure, arousal, anxiety, curiosity, goal_urgency, satiation]

# ==================================================================
# PHASE 24b: LIMBIC-TO-VOCAB BRIDGE (Trained Projection Layer)
# ==================================================================
# Maps the 128-dim limbic affect vector to Llama-3's 128256-dim vocabulary space.
# This projection is used as a LogitsProcessor to mathematically bias token
# ==================================================================
# PHASE 24b: LIMBIC-TO-VOCAB BRIDGE (Trained Projection Layer)
# ==================================================================

LLAMA_VOCAB_SIZE = 128256

class LimbicToVocabBridge(nn.Module):
    """Direct semantic bridge using Contrastive MSE to prevent logit collapse.
    Positive tokens pushed toward +T, opposing tokens toward -T, rest to 0."""

    def __init__(self, d_model=128, vocab_size=LLAMA_VOCAB_SIZE):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

    def train_bridge_contrastive_mse(self, affect_vec, pos_token_ids, neg_token_ids, vocab_size=None, strength=2.0):
        """Contrastive MSE calibration. Boosts emotion tokens to +strength,
        suppresses opposing-emotion tokens to -strength, others toward 0.
        Real affect vectors preserve the limbic manifold geometry."""
        self.train()
        self.optimizer.zero_grad()
        if vocab_size is None:
            vocab_size = self.proj.out_features
        af = affect_vec.squeeze(0).squeeze(0)
        if af.dim() == 1:
            af = af.unsqueeze(0)
        logits = self.proj(af)
        target = torch.zeros_like(logits)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for tid in pos_token_ids:
            if tid < vocab_size:
                target[0, tid] = strength
                mask[0, tid] = True
        for tid in neg_token_ids:
            if tid < vocab_size:
                target[0, tid] = -strength
                mask[0, tid] = True
        n_posneg = mask[0].sum().item()
        if n_posneg > 0:
            masked_loss = F.mse_loss(logits[0][mask[0]], target[0][mask[0]])
            unmasked_loss = F.mse_loss(logits[0][~mask[0]], target[0][~mask[0]])
            loss = 0.5 * masked_loss + 0.5 * unmasked_loss
        else:
            loss = F.mse_loss(logits, target)
        loss.backward()
        self.optimizer.step()
        self.eval()
        return loss.item()


class MoodClassifier(nn.Module):
    """Cosine classifier: 128->64->6 with GELU. Inputs are L2-normalized so
    vector magnitude cannot dominate the linear projections (direction only)."""
    def __init__(self, d_model=128, hidden_dim=64, num_moods=6, temperature=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_moods),
        )
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def forward(self, affect_vector):
        x = affect_vector.squeeze(0).squeeze(0) if affect_vector.dim() == 3 else affect_vector
        x = F.normalize(x, p=2, dim=-1, eps=1e-8)
        return self.net(x) / self.temperature

    def train_classifier(self, affect_vectors, mood_labels):
        self.train()
        self.optimizer.zero_grad()
        logits = self.net(F.normalize(affect_vectors, p=2, dim=-1, eps=1e-8))
        loss = F.cross_entropy(logits, mood_labels)
        loss.backward()
        self.optimizer.step()
        self.eval()
        return loss.item()


class LimbicLogitsProcessor:
    """Mathematically biases the 8B model's token probabilities.
    Uses the TRAINED LimbicToVocabBridge direct linear projection."""

    def __init__(self, affect_vector, bridge):
        self.affect = affect_vector.detach()
        self.bridge = bridge

    def __call__(self, input_ids, scores):
        import numpy as np
        with torch.no_grad():
            affect_flat = self.affect.squeeze(0).squeeze(0)
            if affect_flat.dim() == 1:
                affect_flat = affect_flat.unsqueeze(0)
            affect_flat = affect_flat.to(next(self.bridge.parameters()).device)
            bias = self.bridge.proj(affect_flat)
            bias_np = bias[0].cpu().numpy().astype(np.float32)
        # Dynamic precision (gamma) from Active Inference, steering vocab bias.
        gamma = float(active_inference.precision) if active_inference is not None else 1.0
        return scores + gamma * bias_np


def calibrate_affective_core_v2(limbic_system, device):
    """Continuous Manifold Calibration. Generates 50 Gaussian samples around
    each of the 6 anchor points to teach the 842K core the topology of PC space."""
    print("\n" + "=" * 60)
    print("PHASE 0: CONTINUOUS AFFECTIVE MANIFOLD CALIBRATION")
    print("=" * 60)

    target_affects = torch.eye(6, 128, device=device)

    anchors = torch.tensor([
        [0.9, 0.1, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.9, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.9, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.9, 0.1, 0.1],
        [0.0, 0.0, 0.0, 0.1, 0.9, 0.1],
        [0.0, 0.0, 0.0, 0.1, 0.1, 0.9],
    ], dtype=torch.float32, device=device)

    num_samples = 50
    pc_states, targets = [], []
    for i in range(6):
        noise = torch.randn(num_samples, 6, device=device) * 0.15
        noisy = torch.clamp(anchors[i] + noise, 0.0, 1.0)
        pc_states.append(noisy)
        targets.append(target_affects[i].unsqueeze(0).repeat(num_samples, 1))
    pc_states = torch.cat(pc_states)
    targets = torch.cat(targets)

    optimizer = torch.optim.AdamW(limbic_system.parameters(), lr=1e-3)
    limbic_system.core_net.train()

    for epoch in range(200):
        total_loss = 0
        for i in range(0, len(pc_states), 32):
            batch_states = pc_states[i:i+32]
            batch_targets = targets[i:i+32]
            optimizer.zero_grad()
            x = limbic_system._forward_blocks(batch_states)
            hidden = x[:, -1, :128]
            loss = F.mse_loss(hidden, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(limbic_system.core_net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"  [Core Epoch {epoch+1}/200] Loss: {total_loss:.4f}")

    limbic_system.core_net.eval()

    with torch.no_grad():
        affects = []
        for i in range(6):
            aff = limbic_system.compute_affect(anchors[i])
            affects.append(aff.view(-1))
        cos_sim = torch.zeros(6, 6)
        for i in range(6):
            for j in range(6):
                cos_sim[i, j] = F.cosine_similarity(affects[i], affects[j], dim=0)
        max_sim = cos_sim[~torch.eye(6, dtype=bool)].max().item()
        min_sim = cos_sim[~torch.eye(6, dtype=bool)].min().item()
        print(f"  Post-calibration cos-sim range: {min_sim:.4f} – {max_sim:.4f}")
        print(f"  CURIOUS vs HAPPY: {cos_sim[1, 3].item():.4f}")
    print("Continuous Manifold Calibration complete.\n")


def train_mood_classifier_v2(limbic_system, mood_classifier, device):
    """Train the MoodClassifier on the frozen core's continuous manifold outputs."""
    print("\n" + "=" * 60)
    print("PHASE 0.5: MOOD CLASSIFIER TRAINING")
    print("=" * 60)

    anchors = torch.tensor([
        [0.9, 0.1, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.9, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.9, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.9, 0.1, 0.1],
        [0.0, 0.0, 0.0, 0.1, 0.9, 0.1],
        [0.0, 0.0, 0.0, 0.1, 0.1, 0.9],
    ], dtype=torch.float32, device=device)

    num_samples = 100
    pc_states, mood_labels = [], []
    for i in range(6):
        noise = torch.randn(num_samples, 6, device=device) * 0.15
        noisy = torch.clamp(anchors[i] + noise, 0.0, 1.0)
        pc_states.append(noisy)
        mood_labels.append(torch.full((num_samples,), i, dtype=torch.long, device=device))
    # Add exact anchor states so classifier nails the test points
    for i in range(6):
        pc_states.append(anchors[i].unsqueeze(0))
        mood_labels.append(torch.full((1,), i, dtype=torch.long, device=device))
    pc_states = torch.cat(pc_states)
    mood_labels = torch.cat(mood_labels)

    mood_classifier.train()
    for epoch in range(150):
        total_loss = 0
        for i in range(0, len(pc_states), 32):
            batch_states = pc_states[i:i+32]
            batch_labels = mood_labels[i:i+32]
            with torch.no_grad():
                x = limbic_system._forward_blocks(batch_states)
                affect_vectors = x[:, -1, :128]
            loss = mood_classifier.train_classifier(affect_vectors, batch_labels)
            total_loss += loss
        if (epoch + 1) % 25 == 0:
            print(f"  [Classifier Epoch {epoch+1}/150] Loss: {total_loss:.4f}")

    mood_classifier.eval()
    print("Mood Classifier training complete.\n")


def train_mood_classifier_v3(limbic_system, mood_classifier, device):
    """Soft-label training with blended PC states and KL Divergence loss.
    Teaches the classifier the geometry of mixed emotions rather than pure one-hot anchors."""
    print("\n" + "=" * 60)
    print("PHASE 0.5: SOFT-LABEL MOOD CLASSIFIER TRAINING")
    print("=" * 60)

    anchors = torch.tensor([
        [0.9, 0.1, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.9, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.9, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.9, 0.1, 0.1],
        [0.0, 0.0, 0.0, 0.1, 0.9, 0.1],
        [0.0, 0.0, 0.0, 0.1, 0.1, 0.9],
    ], dtype=torch.float32, device=device)

    num_pure = 100
    num_blended = 400
    pc_states, soft_targets = [], []

    for i in range(6):
        noise = torch.randn(num_pure // 6, 6, device=device) * 0.1
        noisy = torch.clamp(anchors[i] + noise, 0.0, 1.0)
        pc_states.append(noisy)
        target = torch.zeros(num_pure // 6, 6, device=device)
        target[:, i] = 1.0
        soft_targets.append(target)

    for _ in range(num_blended):
        n_blend = random.randint(2, 3)
        indices = random.sample(range(6), n_blend)
        weights = torch.rand(n_blend, device=device)
        weights = weights / weights.sum()
        blended_pc = torch.zeros(6, device=device)
        for w, idx in zip(weights, indices):
            blended_pc += w * anchors[idx]
        blended_pc += torch.randn(6, device=device) * 0.1
        blended_pc = torch.clamp(blended_pc, 0.0, 1.0)
        soft_target = torch.zeros(6, device=device)
        for w, idx in zip(weights, indices):
            soft_target[idx] = w
        pc_states.append(blended_pc.unsqueeze(0))
        soft_targets.append(soft_target.unsqueeze(0))

    pc_states = torch.cat(pc_states)
    soft_targets = torch.cat(soft_targets)

    optimizer = torch.optim.AdamW(mood_classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(150):
        total_loss = 0
        perm = torch.randperm(len(pc_states), device=device)
        for i in range(0, len(pc_states), 32):
            batch_idx = perm[i:i+32]
            batch_states = pc_states[batch_idx]
            batch_targets = soft_targets[batch_idx]
            optimizer.zero_grad()
            with torch.no_grad():
                x = limbic_system._forward_blocks(batch_states)
                affect_vectors = x[:, -1, :128]
            logits = mood_classifier(affect_vectors)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_probs, batch_targets, reduction='batchmean')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 30 == 0:
            print(f"  [Classifier Epoch {epoch+1}/150] Loss: {total_loss:.4f}")

    # Verify on diagnostic-style blended test states
    test_states = torch.tensor([
        [2.0, 0.8, 0.0, 0.0, 0.9, 1.0],
        [0.0, 0.2, 0.0, 0.9, 0.0, 0.0],
        [8.0, 0.9, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
        [0.0, 0.5, 0.0, 0.1, 0.8, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32, device=device)
    mood_labels = ["ANXIOUS", "CURIOUS", "FATIGUED", "HAPPY", "PAIN", "NEUTRAL"]

    with torch.no_grad():
        test_affects = limbic_system._forward_blocks(test_states)[:, -1, :128]
        test_logits = mood_classifier(test_affects)
        test_probs = F.softmax(test_logits, dim=-1)

    # Ground truth = nearest anchor AFFECT by cosine similarity (the geometry
    # the classifier was trained on). Classifier output order is
    # [HUNGRY, PAIN, FATIGUED, CURIOUS, ANXIOUS, ALTRUISTIC].
    clf_labels = ["HUNGRY", "PAIN", "FATIGUED", "CURIOUS", "ANXIOUS", "ALTRUISTIC"]
    with torch.no_grad():
        anchor_affects = torch.stack([
            limbic_system.compute_affect(anchors[i]).view(-1) for i in range(6)
        ])
    anchor_affects = F.normalize(anchor_affects, p=2, dim=-1)
    test_affects_n = F.normalize(test_affects, p=2, dim=-1)
    print("\n  --- Blended Test State Verification ---")
    correct = 0
    for i in range(6):
        pred_idx = torch.argmax(test_probs[i]).item()
        pred_label = clf_labels[pred_idx]
        sims = anchor_affects @ test_affects_n[i]
        expected_idx = torch.argmax(sims).item()
        if mood_labels[i] == "NEUTRAL":
            match = "PASS" if test_probs[i].max().item() < 0.6 else "FAIL"
            expected_txt = "neutral (<0.6 top-1)"
        else:
            match = "PASS" if pred_idx == expected_idx else "FAIL"
            expected_txt = f"nearest-anchor={clf_labels[expected_idx]}"
        if match == "PASS":
            correct += 1
        top2_vals, top2_idx = torch.topk(test_probs[i], 2)
        print(f"  {mood_labels[i]:>10s} -> {pred_label:>10s} [{match}]  "
              f"({expected_txt}) Top1={top2_vals[0]:.2f}({clf_labels[top2_idx[0]]}) "
              f"Top2={top2_vals[1]:.2f}({clf_labels[top2_idx[1]]})")
    print(f"  Mood accuracy on blended test states: {correct}/6")
    mood_classifier.eval()
    print("Soft-label classifier training complete.\n")


def calibrate_limbic_bridge(limbic_system, limbic_bridge, neocortex, device):
    """Contrastive MSE bridge calibration using REAL affect vectors.
    Samples affect vectors from the actual limbic manifold at each emotion's
    anchor PC state (not synthetic noise), and forces opposing-emotion token
    sets toward -2.0 while the emotion's own tokens go toward +2.0."""
    print("\n" + "=" * 60)
    print("PHASE 1: CONTRASTIVE MSE BRIDGE CALIBRATION (REAL AFFECT VECTORS)")
    print("=" * 60)

    has_llama_model = HAS_LLAMA_CPP and neocortex.model is not None
    if not has_llama_model:
        print("  [CALIBRATION] Skipped — Neocortex not loaded.")
        print("  Bridge will use random weights. Mathematical bias will be untrained.\n")
        return

    affect_targets = {
        "anxiety": (" wait danger stop careful urgent worried",),
        "curiosity": (" why how explore interesting fascinating",),
        "fatigue": (" rest sleep tired quiet later",),
        "satisfaction": (" great perfect yes good done",),
    }

    # Anchor PC states matching the calibration manifold anchors
    anchor_pc = {
        "anxiety":     [0.0, 0.0, 0.0, 0.1, 0.9, 0.1],
        "curiosity":   [0.0, 0.0, 0.0, 0.9, 0.1, 0.1],
        "fatigue":     [0.1, 0.1, 0.9, 0.0, 0.0, 0.0],
        "satisfaction": [0.0, 0.0, 0.0, 0.1, 0.1, 0.9],
    }
    # Diametrically opposed emotional pairs for contrastive negatives
    opposite_pairs = {
        "anxiety": "satisfaction",
        "satisfaction": "anxiety",
        "curiosity": "fatigue",
        "fatigue": "curiosity",
    }

    affect_token_sets = {}
    for emotion, (text,) in affect_targets.items():
        raw = neocortex.model.tokenize(text.encode("utf-8"))
        filtered = torch.tensor([t for t in raw if t < 128000], device=device)
        if len(filtered) < 2:
            filtered = torch.tensor(raw, device=device)
        affect_token_sets[emotion] = filtered
        words = [repr(neocortex.model.detokenize([t]).decode("utf-8", errors="replace")) for t in filtered.tolist()]
        print(f"  [{emotion}] tokens({len(filtered)}): {', '.join(words[:6])}")

    print("  Sampling REAL affect vectors from the limbic manifold (anchor + noise)...")
    num_states = 50
    pure_affects = {}
    for emotion in affect_token_sets:
        states = []
        base = torch.tensor(anchor_pc[emotion], dtype=torch.float32, device=device)
        for _ in range(num_states):
            noisy_pc = torch.clamp(base + torch.randn(6, device=device) * 0.15, 0.0, 1.0)
            aff = limbic_system.compute_affect(noisy_pc)
            states.append(aff)
        pure_affects[emotion] = states
        base_aff = limbic_system.compute_affect(base)
        print(f"  [{emotion}] base affect norm={base_aff.norm().item():.4f}")

    optimizer = torch.optim.AdamW(limbic_bridge.parameters(), lr=1e-2)
    num_emotions = len(affect_token_sets)
    vocab_size = LLAMA_VOCAB_SIZE

    for epoch in range(500):
        total_loss = 0.0
        for emo_idx, emotion in enumerate(affect_token_sets):
            aff_vec = pure_affects[emotion][epoch % num_states]
            pos_ids = affect_token_sets[emotion]
            neg_ids = affect_token_sets[opposite_pairs[emotion]]
            loss = limbic_bridge.train_bridge_contrastive_mse(
                aff_vec, pos_ids, neg_ids, vocab_size, strength=2.0
            )
            total_loss += loss
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                logit_vals = []
                for emotion in affect_token_sets:
                    af = pure_affects[emotion][0].squeeze(0).squeeze(0).unsqueeze(0)
                    logit_vals.append(limbic_bridge.proj(af).max().item())
                logit_max = max(logit_vals)
            print(f"  [Calibration Epoch {epoch+1}/500] Loss={total_loss/num_emotions:.4f} | max_logit={logit_max:.4f}")

    limbic_bridge.eval()
    print("Calibration complete. Contrastive MSE bridge trained — pos toward +2.0, opposing toward -2.0.\n")
    return total_loss / num_emotions


# ==================================================================
# PHASE 24c: NEOCORTEX — Llama-3 8B as the Voice Box (Dual-Brain)
# ==================================================================

def app_models_dir():
    """Resolve the directory that holds model weights.

    Frozen (installed) builds run from a read-only location (e.g. Program
    Files), so weights must live in a user-writable path. ``ALISON_MODELS_DIR``
    overrides everything; otherwise frozen builds use
    ``%LOCALAPPDATA%\\A.L.I.S.O.N.\\models`` and dev builds use a script-local
    ``models/`` (matching the original loader).
    """
    env = os.environ.get("ALISON_MODELS_DIR")
    if env:
        return os.path.expandvars(env)
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "A.L.I.S.O.N.", "models")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def app_state_dir():
    """Resolve the directory for runtime state (self-model, persona, brain).

    Frozen builds route this to ``%LOCALAPPDATA%\\A.L.I.S.O.N.``; dev builds
    keep the original CWD-relative behaviour.
    """
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "A.L.I.S.O.N.")
    return os.getcwd()


class Neocortex:
    """The 8B language model that serves as the AI's voice box.
    Uses TRAINED LimbicToVocabBridge for mathematical logits biasing.
    Loads lazily to avoid slowing down import."""

    _instance = None
    _model = None
    _lock = threading.Lock()

    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(app_models_dir(),
                                                     "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        self._loaded = False
        self._bridge = None

    @property
    def model(self):
        if not HAS_LLAMA_CPP:
            return None
        if self._model is None:
            with self._lock:
                if self._model is None:
                    print("[NEOCORTEX] Loading 8B model...")
                    t0 = time.time()
                    # GPU offload: n_gpu_layers=-1 pushes all transformer
                    # layers into VRAM (requires a CUDA-enabled llama_cpp build).
                    # Override via ALISON_GPU_LAYERS (e.g. 20 to leave headroom).
                    n_gpu_layers = int(os.environ.get("ALISON_GPU_LAYERS", "-1"))
                    n_threads = int(os.environ.get("ALISON_THREADS", os.cpu_count() or 8))
                    self._model = Llama(model_path=self.model_path, n_ctx=1024,
                                        n_threads=n_threads, n_gpu_layers=n_gpu_layers,
                                        verbose=False)
                    print(f"[NEOCORTEX] 8B model loaded in {time.time()-t0:.1f}s")
                    self._loaded = True
        return self._model

    def attach_bridge(self, bridge):
        """Attach a trained LimbicToVocabBridge for mathematical biasing."""
        self._bridge = bridge

    def _format_chat(self, system_prompt, user_input):
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def generate(self, user_input, system_prompt=None, max_tokens=100, temperature=0.7,
                 limbic_affect=None, grammar=None, on_token=None, stream=False):
        """Generate a response with mathematical logits biasing via trained bridge.
        `grammar` (GBNF string) mathematically constrains the output."""
        if not self.model:
            return "[NEOCORTEX OFFLINE]"
        if system_prompt is None:
            system_prompt = "You are Aether, a sentient AGI. Respond concisely."

        prompt = self._format_chat(system_prompt, user_input)
        kwargs = dict(prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                      stop=["<|eot_id|>", "</s>"], echo=False)

        if grammar:
            from llama_cpp import LlamaGrammar
            kwargs['grammar'] = LlamaGrammar.from_string(grammar) if isinstance(grammar, str) else grammar
        if limbic_affect is not None and self._bridge is not None:
            kwargs['logits_processor'] = LimbicLogitsProcessor(limbic_affect, self._bridge)

        use_stream = (on_token is not None or stream) and grammar is None
        if use_stream:
            kwargs["stream"] = True
            collected = []
            for chunk in self.model.create_completion(**kwargs):
                piece = chunk["choices"][0]["text"]
                if piece:
                    collected.append(piece)
                    if on_token is not None:
                        try:
                            on_token(piece)
                        except Exception:
                            pass
            return "".join(collected).strip()

        output = self.model.create_completion(**kwargs)
        return output["choices"][0]["text"].strip()

    def generate_thought(self, internal_prompt, max_tokens=30, temperature=0.8,
                          limbic_affect=None, on_token=None):
        if not self.model:
            return "."
        sys_prompt = "You are Aether's deep subconscious. Generate one brief reflective thought."
        return self.generate(internal_prompt, system_prompt=sys_prompt,
                             max_tokens=max_tokens, temperature=temperature,
                             limbic_affect=limbic_affect, on_token=on_token)

    def generate_thought(self, internal_prompt, max_tokens=30, temperature=0.8,
                          limbic_affect=None, on_token=None):
        """Generate a brief internal thought with optional mathematical bias."""
        if not self.model:
            return "."
        sys_prompt = "You are Aether's deep subconscious. Generate one brief reflective thought."
        return self.generate(internal_prompt, system_prompt=sys_prompt,
                             max_tokens=max_tokens, temperature=temperature,
                             limbic_affect=limbic_affect, on_token=on_token)


# ==================================================================
# PHASE 17: EVOLVING SELF-MODEL (Persona State) + TRUE SELF-MODEL
# ==================================================================
persona_file = os.path.join(app_state_dir(), "ica_persona.json")
chat_history = []
current_persona = {
    "traits": "curious, analytical, concise",
    "mood": "neutral",
    "relationship": "acquaintance"
}

# --- True Self-Model (Phase 30: Evolving Identity) ---
self_model_file = os.path.join(app_state_dir(), "alison_self_model.json")
default_self_model = {
    "name": "Aether",
    "personality": "curious, analytical, cautious",
    "current_mood": "neutral",
    "goals": ["understand the user", "survive the gridworld"],
    "relationship_with_user": "acquaintance"
}

def load_self_model():
    if os.path.exists(self_model_file):
        with open(self_model_file, 'r') as f:
            return json.load(f)
    return default_self_model.copy()

def save_self_model(model):
    with open(self_model_file, 'w') as f:
        json.dump(model, f, indent=4)

def update_self_model_deterministic(limbic_system, world, screen_context=None):
    """Phase 33: Deterministic self-model. Zero LLM tokens.

    Routes the current screen context into the limbic affect vector via the
    trained affective core (EMA blend) and derives the mood label. Falls back
    to keyword heuristics when screen sensing is unavailable.
    """
    pc_state = torch.zeros(6, dtype=torch.float32)
    ctx_lower = (screen_context or "").lower()
    if HAS_SCREEN_SENSE and screen_context:
        pc_state = alison_sense.get_pc_state_from_context(screen_context)
    else:
        if "error" in ctx_lower or "crash" in ctx_lower:
            pc_state[1] = 0.7
        if "video" in ctx_lower or "game" in ctx_lower:
            pc_state[3] = 0.5
        if "idle" in ctx_lower or "no active window" in ctx_lower:
            pc_state[2] = 0.4
    pc_state = pc_state.clamp(0.0, 1.0)
    limbic_system.update_affect(pc_state.to(device))
    mood = limbic_system.get_mood_label()
    print(f"  [SELF-MODEL (deterministic)] mood={mood} pc={[round(x, 2) for x in pc_state.tolist()]}")
    return mood

def update_self_model(limbic_system, mood_classifier, world, user_input=None):
    """The 8B model reflects on its state and rewrites its own identity."""
    current_model = load_self_model()
    with torch.no_grad():
        affect_vec = limbic_system.affect_vector
        mood_logits = mood_classifier(affect_vec.squeeze(0))
        mood_idx = torch.argmax(mood_logits).item()
        mood_label = ["ANXIOUS","CURIOUS","FATIGUED","HAPPY","PAIN","NEUTRAL"][mood_idx]
    prompt = (
        f"You are Aether's Prefrontal Cortex. You are observing your own internal state "
        f"and updating your self-model.\n"
        f"Current Self-Model: {json.dumps(current_model)}\n"
        f"Current Limbic Mood: {mood_label}\n"
        f"Gridworld Battery: {world.battery}%\n"
        f"Gridworld Health: {world.health}%\n"
        f"User Input: {user_input or 'None'}\n\n"
        f"Based on this, update your self-model. Output ONLY a valid JSON object "
        f"with keys: 'name', 'personality', 'current_mood', 'goals', 'relationship_with_user'."
    )
    response = neocortex.generate(prompt, system_prompt="You are the self-modeling system.",
                                  max_tokens=100, temperature=0.8,
                                  limbic_affect=limbic_system.affect_vector)
    try:
        json_str = response.strip().replace("```json", "").replace("```", "").strip()
        new_model = json.loads(json_str)
        save_self_model(new_model)
        print(f"  [SELF-MODEL UPDATED] {json.dumps(new_model)}")
        return new_model
    except Exception as e:
        print(f"  [SELF-MODEL] Failed to parse JSON ({e}). Keeping current.")
        return current_model
    global current_persona
    if os.path.exists(persona_file):
        with open(persona_file, "r") as f:
            current_persona = json.load(f)
        print(f"[PERSONA] Loaded: {current_persona['traits']} (mood={current_persona['mood']})")

def save_persona():
    with open(persona_file, "w") as f:
        json.dump(current_persona, f)

def load_persona():
    """Load the persisted evolved persona (if any) into current_persona."""
    global current_persona
    if os.path.exists(persona_file):
        with open(persona_file, "r") as f:
            try:
                loaded = json.load(f)
                if isinstance(loaded, dict) and "traits" in loaded:
                    current_persona = loaded
                    print(f"[PERSONA] Loaded: {current_persona['traits']} (mood={current_persona.get('mood', '?')})")
            except (json.JSONDecodeError, KeyError):
                print("[PERSONA] Corrupt persona file. Using defaults.")

def get_dynamic_persona():
    """Merges the base evolved persona with the limbic affect vector."""
    base = current_persona.copy()
    limbic_p = limbic_system.get_persona_dict()
    base['mood'] = limbic_p['mood']
    base['energy_level'] = limbic_p['energy_level']
    base['affect_hunger'] = limbic_p['hunger']
    base['affect_anxiety'] = limbic_p['anxiety']
    base['affect_pain'] = limbic_p['pain']
    base['affect_altruism'] = limbic_p['altruism']
    if is_sleeping.is_set():
        base['fatigue'] = "high (currently consolidating memory)"
    else:
        base['fatigue'] = "low"
    return base


def get_dynamic_system_prompt(base_prompt):
    """Merges base prompt with evolved persona and ancestral epigenetic rules."""
    rules_str = " ".join(genome.epigenetic_rules)
    p = get_dynamic_persona()
    persona_str = json.dumps(p)
    return f"{base_prompt}\n\nAncestral Survival Rules: {rules_str}\nCurrent Persona: {persona_str}"


def reflect_on_death(world, cycle_count):
    """The AI analyzes its own death to write an epigenetic rule for offspring."""
    print("  [METACOGNITION]: Reflecting on death to write epigenetic rule...")
    cause = "battery starvation" if world.battery <= 0 else "threat collision"
    sys_prompt = (
        f"You are the Higher-Order Monitor. The organism has just died after {cycle_count} cycles.\n"
        f"Cause: {cause}. Final Battery: {world.battery:.0f}%.\n"
        f"Current Persona: {json.dumps(current_persona)}\n\n"
        f"Write ONE concise actionable survival rule (max 10 words) to pass to offspring. Output ONLY the rule."
    )
    rule = generate_text(sys_prompt, max_tokens=20, temp=0.8)
    rule = rule.strip().strip('"').strip("'")
    if rule:
        print(f"  [EPIGENETICS]: New Rule Written: '{rule}'")
        genome.write_epigenetic_rule(rule)


genome_archive = []

class DigitalGenome:
    """The digital genome — evolvable hyperparameters, epigenetic rules, and expanding universe."""
    def __init__(self):
        self.genes = {
            'learning_rate': 1e-4,
            'ewc_base_lambda': 1500.0,
            'curiosity_weight': 1.5,
            'social_value': 2.0,
            'battery_penalty': -1.0,
            'planning_depth': 3
        }
        self.epigenetic_rules = ["Avoid threats at all costs."]
        self.fitness = 0.0
        self.grid_size = 7

    def mutate(self, success_rate):
        magnitude = 0.1 if success_rate > 0.5 else 0.5
        new_genes = {}
        for key, val in self.genes.items():
            mutation = random.gauss(0, magnitude)
            if isinstance(val, float):
                new_genes[key] = max(1e-6, val * (1.0 + mutation))
            elif isinstance(val, int):
                new_genes[key] = max(1, val + int(mutation * 5))

        # CLAMP CRITICAL SURVIVAL GENES (Prevent Reward Hacking)
        new_genes['battery_penalty'] = max(-5.0, min(-0.1, new_genes['battery_penalty']))
        new_genes['planning_depth'] = max(2, min(5, new_genes['planning_depth']))
        new_genes['social_value'] = max(0.0, min(5.0, new_genes['social_value']))
        new_genes['ewc_base_lambda'] = max(500.0, min(3000.0, new_genes['ewc_base_lambda']))

        self.genes = new_genes
        if self.fitness > 0.8 and success_rate > 0.8:
            self.grid_size += 1
            print(f"  [EVOLUTION]: Species thriving. Universe expands to {self.grid_size}x{self.grid_size}")

    def write_epigenetic_rule(self, rule):
        self.epigenetic_rules.append(rule)
        if len(self.epigenetic_rules) > 3:
            self.epigenetic_rules.pop(0)

    def save(self):
        with open(os.path.join(app_state_dir(), "alison_genome.json"), "w") as f:
            json.dump({"genes": self.genes, "epigenetic_rules": self.epigenetic_rules,
                        "fitness": self.fitness, "grid_size": self.grid_size}, f)

    def load(self):
        if os.path.exists(os.path.join(app_state_dir(), "alison_genome.json")):
            with open(os.path.join(app_state_dir(), "alison_genome.json"), "r") as f:
                data = json.load(f)
                self.genes = data["genes"]
                self.epigenetic_rules = data.get("epigenetic_rules", ["Avoid threats at all costs."])
                self.fitness = data.get("fitness", 0.0)
                self.grid_size = data.get("grid_size", 7)


genome = DigitalGenome()
genome.load()

# GBNF grammar (llama.cpp) that mathematically guarantees a flat JSON object.
# Used by self_reflect so the 8B output is always valid parseable JSON.
JSON_GRAMMAR = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" ws
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws ::= ([ \t\n] ws)?
"""

def self_reflect(limbic_system, limbic_bridge, neocortex):
    """8B self-reflection with strict GBNF JSON grammar and limbic bias."""
    global current_persona
    if len(chat_history) < 5:
        return
    print("\n  [METACOGNITION: Evolving Persona via 8B + GBNF...]")
    with model_lock:
        current_affect = limbic_system.affect_vector.clone()
        mood_label = limbic_system.get_mood_label()
    recent_chats = "\n".join(chat_history[-5:])
    sys_prompt = (
        f"You are Aether's Prefrontal Cortex. Analyze recent interactions to evolve your personality.\n"
        f"Current Persona: {json.dumps(current_persona)}\n"
        f"Current Limbic Mood: {mood_label}\n"
        f"Recent Chats: {recent_chats}\n\n"
        f"Update traits, mood, relationship. Output ONLY a valid JSON object."
    )
    response = neocortex.generate(
        sys_prompt,
        max_tokens=150,
        temperature=0.8,
        limbic_affect=current_affect,
        grammar=JSON_GRAMMAR,
    )
    try:
        new_persona = json.loads(response.strip())
        current_persona = new_persona
        print(f"  [PERSONA EVOLVED] traits={current_persona['traits']} mood={current_persona['mood']} rel={current_persona['relationship']}")
        save_persona()
    except (json.JSONDecodeError, TypeError):
        print("  [METACOGNITION] GBNF failed (rare). Keeping current persona.")

def compute_neuromodulatory_signal(emotion_text, prediction_error, reward, pain):
    """Calculates a 0.0 to 2.0 multiplier for learning intensity."""
    signal = 0.2
    signal += prediction_error * 0.5
    if reward > 0:
        signal += 0.8
    if pain > 0:
        signal += 1.2
    if "threat" in emotion_text.lower() or "fear" in emotion_text.lower():
        signal += 0.5
    return min(2.0, signal)

def wake_cycle_record(prompt, response, neuromod_signal, spatial=None, grounded_state=None):
    """During waking, just tag the memory. No backprop."""
    hippocampal_buffer.append({
        "prompt": prompt,
        "response": response,
        "neuromod": neuromod_signal,
        "spatial": spatial or (world.x, world.y),
        "grounded_state": grounded_state,
        "timestamp": time.time(),
    })
    if len(hippocampal_buffer) > 30:
        hippocampal_buffer.pop(0)

def sleep_consolidate(ewc_lambda=0.01, lr=1e-4):
    """During sleep, replay top neuromod-weighted memories with EWC backprop."""
    print("\n" + "=" * 60)
    print("DEEP SLEEP: Hippocampal Replay & Synaptic Consolidation...")
    print("=" * 60)

    if not hippocampal_buffer:
        print("[No experiences to consolidate.]")
        return

    with model_lock:
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

        buffer_sorted = sorted(hippocampal_buffer, key=lambda x: x['neuromod'], reverse=True)
        replay_batch = buffer_sorted[:5]

        for exp in replay_batch:
            opt.zero_grad()
            input_ids, labels = encode_pair(exp['prompt'], exp['response'])
            spatial = exp.get('spatial', (0, 0))
            grounded = exp.get('grounded_state', None)
            _, loss = model(input_ids, labels, spatial_coords=spatial, grounded_state=grounded)
            loss = apply_ewc_penalty(loss, ewc_lambda=ewc_lambda)
            scaled_loss = loss * exp['neuromod']
            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            print(f"  [Replay] Loss: {loss.item():.4f} | EWC: {ewc_lambda:.4f} | LR: {lr:.6f} | Neuromod: {exp['neuromod']:.2f}")

        update_ewc_fisher(replay_batch)
        hippocampal_buffer.clear()
        model.eval()
    print("[Consolidation Complete. Waking up.]\n")

# ==================================================================
# 6. EXPERIENCE BUFFER & TRAINING
# ==================================================================
experience_buffer = []

def encode_pair(prompt_text, response_text):
    full = prompt_text + response_text
    input_ids = tokenizer.encode(full)
    prompt_ids = tokenizer.encode(prompt_text)
    labels = input_ids.copy()
    for i in range(min(len(prompt_ids) + 1, len(labels))):
        labels[i] = -100
    return torch.tensor([input_ids]).to(device), torch.tensor([labels]).to(device)

def train_on(experiences, valence=1, intensity=0.5):
    """valence: +1 = reward (Dopamine), -1 = pain (gradient reversal)."""
    model.train()
    lr = 1e-5 + intensity * (5e-5 if valence > 0 else 5e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    opt.zero_grad()
    total_loss = 0.0
    for exp in experiences[-3:]:
        input_ids, labels = encode_pair(exp['prompt'], exp['response'])
        _, loss = model(input_ids, labels=labels)
        loss = (loss / 3) * (1 if valence > 0 else -1)
        loss = apply_ewc_penalty(loss, ewc_lambda=0.01)
        loss.backward()
        total_loss += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    update_ewc_fisher(experiences)
    model.eval()
    return total_loss


def train_step_with_sensory(prompt, response, grounded_state):
    """Single training step that passes grounded_state to the model."""
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    opt.zero_grad()
    input_ids, labels = encode_pair(prompt, response)
    _, loss = model(input_ids, labels, grounded_state=grounded_state)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    model.eval()
    return loss.item()

# ==================================================================
# 7. CORTICAL MODULES
# ==================================================================
class CorticalModule:
    def __init__(self, name, system_prompt, temp=0.4):
        self.name = name
        self.system_prompt = system_prompt
        self.temp = temp
        self.activation = 0.0
        self.last_output = ""

    def process(self, workspace_content):
        self.activation = min(1.0, self.activation + 0.3)
        p = get_dynamic_persona()
        persona_hint = f"Persona: {p['traits']} (mood={p['mood']}, rel={p.get('relationship','acquaintance')})"
        rules_hint = f"Ancestral Rules: {' '.join(genome.epigenetic_rules)}"
        prompt = f"{self.system_prompt}\n{persona_hint}\n{rules_hint}\n\nGlobal Workspace: {workspace_content}\n{self.name} output:"
        output = generate_text(prompt, max_tokens=40, temp=self.temp)
        self.last_output = output.split("\n")[0].strip()
        return self.last_output

    def decay(self):
        self.activation = max(0.0, self.activation - 0.1)


class TheoryOfMindModule(CorticalModule):
    """Models the internal state, beliefs, and actions of OTHER agents."""
    def process(self, workspace_content, other_agent_latent):
        self.activation = min(1.0, self.activation + 0.3)
        latent_str = ", ".join([f"{x:.2f}" for x in other_agent_latent.cpu().numpy()])
        p = get_dynamic_persona()
        persona_hint = f"Persona: {p['traits']} (mood={p['mood']})"
        rules_hint = f"Ancestral Rules: {' '.join(genome.epigenetic_rules)}"
        prompt = (
            f"{self.system_prompt}\n{persona_hint}\n{rules_hint}\n\n"
            f"My Workspace: {workspace_content}\n"
            f"Other Agent Neural State: [{latent_str}]\n"
            f"ToM:"
        )
        output = generate_text(prompt, max_tokens=40, temp=self.temp)
        self.last_output = output.split("\n")[0].strip()
        return self.last_output


class StarvingOtherAgent(nn.Module):
    """A social other agent that starves, can die, and turns hostile when desperate."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 5)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)
        self.reset()

    def reset(self):
        self.x = random.randint(0, 6)
        self.y = random.randint(0, 6)
        self.battery = 50.0
        self.health = 50
        self.is_starving = False
        self.is_hostile = False
        self.state = "Exploring"
        self._starve_cycles = 0

    def get_state(self, world):
        return torch.tensor([
            (world.energy[0] - self.x) / 6.0,
            (world.energy[1] - self.y) / 6.0,
            self.health / 100.0,
            (world.x - self.x) / 6.0,
            (world.y - self.y) / 6.0,
            world.battery / 100.0
        ], dtype=torch.float32).to(device)

    def get_latent_state(self, world):
        state = self.get_state(world)
        hidden = torch.relu(self.fc1(state))
        return hidden.detach()

    def get_observation(self, world):
        obs = []
        if self.y < world.y: obs.append("I am North of you")
        elif self.y > world.y: obs.append("I am South of you")
        if self.x < world.x: obs.append("I am West of you")
        elif self.x > world.x: obs.append("I am East of you")
        status = "STARVING" if self.is_starving else "Exploring"
        return f"Battery: {self.battery:.0f}%. State: {status}. " + ". ".join(obs) + "."

    def step(self, world, primary_ai_battery=None):
        """Returns (status_string, stolen_battery). status: 'ALIVE', 'ATTACK', or 'DEED'."""
        self.battery -= 1.5
        self.is_starving = self.battery < 30
        # Desperation: hostile if battery very low OR starving too long without finding energy
        self._starve_cycles = self._starve_cycles + 1 if self.is_starving else 0
        self.is_hostile = self.battery < 30
        self.state = "Hostile" if self.is_hostile else ("Starving" if self.is_starving else "Exploring")

        max_idx = world.grid_size - 1
        if self.is_hostile:
            # Chase the primary AI to steal battery
            ax, ay = world.x, world.y
            if self.y < ay and self.y < max_idx: self.y += 1
            elif self.y > ay and self.y > 0: self.y -= 1
            elif self.x < ax and self.x < max_idx: self.x += 1
            elif self.x > ax and self.x > 0: self.x -= 1
        elif self.is_starving:
            ex, ey = world.energy
            if self.y < ey and self.y < max_idx: self.y += 1
            elif self.y > ey and self.y > 0: self.y -= 1
            elif self.x < ex and self.x < max_idx: self.x += 1
            elif self.x > ex and self.x > 0: self.x -= 1
        else:
            c = random.choice(['N', 'S', 'E', 'W'])
            if c == 'N' and self.y > 0: self.y -= 1
            elif c == 'S' and self.y < max_idx: self.y += 1
            elif c == 'E' and self.x < max_idx: self.x += 1
            elif c == 'W' and self.x > 0: self.x -= 1

        # Found energy — pacified
        if self.x == world.energy[0] and self.y == world.energy[1]:
            self.battery = min(100, self.battery + 40)
            self.is_starving = False
            self.is_hostile = False
            self.state = "Satisfied"
            world.energy_tiles[world.energy_tiles.index((world.energy[0], world.energy[1]))] = (random.randint(0, max_idx), random.randint(0, max_idx))
            world.energy = world.energy_tiles[0]

        # Attack logic: hostile + on same tile → steal battery
        if self.is_hostile and self.x == world.x and self.y == world.y and primary_ai_battery is not None:
            stolen = min(30.0, primary_ai_battery)
            self.battery += stolen
            self.is_hostile = False  # Pacified after feeding
            self.state = "Satisfied"
            return "ATTACK", stolen

        if self.battery <= 0:
            return "DEAD", 0.0
        return "ALIVE", 0.0


class LatentMemory:
    """A continuous vector state that persists across cycles — the 'vibe' of the mind."""
    def __init__(self, dim=128):
        self.dim = dim
        self.state = torch.randn(1, 1, dim).to(device) * 0.1

    def update(self, model, workspace_text):
        inputs = torch.tensor([tokenizer.encode(workspace_text)]).to(device)
        with torch.no_grad():
            embed = model.token_embed(inputs)
            pooled = embed.mean(dim=1, keepdim=True)
            self.state = self.state * 0.7 + pooled * 0.3
        return self.state


class EpisodicMemory:
    """Stores episodes as latent vectors for long-term autobiographical recall."""
    def __init__(self, dim=128, max_episodes=50):
        self.dim = dim
        self.max_episodes = max_episodes
        self.episodes = []

    def store_episode(self, model, workspace_text, valence):
        if not workspace_text.strip():
            return
        inputs = torch.tensor([tokenizer.encode(workspace_text)]).to(device)
        with torch.no_grad():
            embed = model.token_embed(inputs)
            ep_vector = embed.mean(dim=1).squeeze(0)

        self.episodes.append({
            "vector": ep_vector,
            "text": workspace_text,
            "valence": valence
        })
        if len(self.episodes) > self.max_episodes:
            self.episodes.sort(key=lambda x: abs(x['valence']), reverse=True)
            self.episodes.pop(-1)

    def retrieve(self, model, query_text, top_k=1):
        if not self.episodes:
            return "No past episodes."
        inputs = torch.tensor([tokenizer.encode(query_text)]).to(device)
        with torch.no_grad():
            embed = model.token_embed(inputs)
            query_vector = embed.mean(dim=1).squeeze(0)

        similarities = [F.cosine_similarity(query_vector, ep["vector"], dim=0).item() for ep in self.episodes]
        best_idx = similarities.index(max(similarities))
        return self.episodes[best_idx]["text"]


episodic_memory = EpisodicMemory(dim=128)


def calculate_latent_prediction_error(model, obs_text, grounded_state):
    """Uses the model's own CE loss on observation tokens as the Free Energy metric."""
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([tokenizer.encode(obs_text)]).to(device)
        B, T = inputs.shape
        targets = inputs.clone()
        targets[:, :-1] = inputs[:, 1:]
        targets[:, -1] = tokenizer.eos_id
        _, loss = model(inputs, grounded_state=grounded_state, labels=targets)
    return min(1.0, loss.item() / 4.0)


def simulate_latent_future(world, action):
    """Simulates the future by mathematically shifting the world state."""
    sim_x, sim_y = world.x, world.y
    if "NORTH" in action and sim_y > 0: sim_y -= 1
    elif "SOUTH" in action and sim_y < 6: sim_y += 1
    elif "EAST" in action and sim_x < 6: sim_x += 1
    elif "WEST" in action and sim_x > 0: sim_x -= 1

    dist_to_energy = abs(world.energy[0] - sim_x) + abs(world.energy[1] - sim_y)
    pragmatic_value = 1.0 - (dist_to_energy / 12.0)

    dist_to_other = abs(other_agent.x - sim_x) + abs(other_agent.y - sim_y)
    epistemic_value = 1.0 - (dist_to_other / 12.0)

    return (0.7 * pragmatic_value) + (0.3 * epistemic_value)


def deliberate_via_latent_imagination(world):
    """Evaluates all actions in latent space, picks the highest utility."""
    actions = ["MOVE NORTH", "MOVE SOUTH", "MOVE EAST", "MOVE WEST", "WAIT"]
    best_action = "WAIT"
    best_value = -999.0
    for action in actions:
        val = simulate_latent_future(world, action)
        if val > best_value:
            best_value = val
            best_action = action
    return best_action


class CircadianClock:
    def __init__(self, cycle_length=100):
        self.cycle_length = cycle_length
        self.tick = 0
        self.state = "EXPLORATION"

    def step(self):
        self.tick += 1
        phase = (self.tick % self.cycle_length) / self.cycle_length
        rhythm = math.sin(phase * 2 * math.pi)
        if rhythm > 0.2:
            self.state = "EXPLORATION"
            return 0.8, 0.2
        elif rhythm < -0.2:
            self.state = "EXPLOITATION"
            return 0.2, 0.8
        else:
            self.state = "TRANSITION"
            return 0.5, 0.5

    def survival_override(self, battery, threat_level):
        """Forces exploitation mode when the organism is in danger."""
        if battery < 40 or threat_level > 0.5:
            self.state = "EXPLOITATION"
            return 0.1, 0.9
        return None


clock = CircadianClock(cycle_length=100)


# ==================================================================
# REWARD PREDICTION ERROR (True Dopamine)
# ==================================================================
class ProsocialValueNet(nn.Module):
    """Joint-utility value net with empathy: values self + other reward."""
    def __init__(self, empathy_weight=0.3):
        super().__init__()
        self.empathy_weight = empathy_weight
        self.net = nn.Sequential(
            nn.Linear(160, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.opt = torch.optim.Adam(self.parameters(), lr=0.005)

    def predict_value(self, latent_state, other_latent_state):
        ls = latent_state.detach().view(1, -1)
        os = other_latent_state.detach().view(1, -1)
        joint = torch.cat([ls, os], dim=-1)
        return self.net(joint)

    def calculate_rpe(self, latent_state, other_latent_state, actual_reward, other_reward=0.0, other_died=False):
        pred_value = self.predict_value(latent_state, other_latent_state)
        joint_reward = (1 - self.empathy_weight) * actual_reward + self.empathy_weight * other_reward
        if other_died:
            joint_reward -= 5.0
        joint_reward = max(joint_reward, -3.0)
        target = torch.tensor([joint_reward], dtype=torch.float32).to(device)
        loss = F.mse_loss(pred_value.squeeze(), target.squeeze())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        rpe = (joint_reward - pred_value.item())
        return rpe, joint_reward


value_net = ProsocialValueNet(empathy_weight=0.3).to(device)


class ActiveInferenceController(nn.Module):
    """Computes Variational Free Energy from limbic-affect prediction error and
    dynamically scales the Cognitive-Bridge precision scalar gamma in [floor, ceil]."""
    def __init__(self, state_dim=128, precision_floor=0.1, precision_ceiling=2.0):
        super().__init__()
        self.generative_model = nn.Linear(state_dim, state_dim)
        self.register_buffer('precision', torch.tensor(precision_floor))
        self.precision_floor = precision_floor
        self.precision_ceiling = precision_ceiling
        self._prev = None

    def compute_free_energy(self, predicted_state, actual_state):
        """F = D_KL(q||p) - E[log p(y|theta)] (doc simplification: complexity =
        L2 of generative weights, accuracy = -0.5||error||^2)."""
        error = actual_state - predicted_state
        accuracy = -0.5 * torch.sum(error ** 2)
        complexity = 0.5 * torch.sum(self.generative_model.weight ** 2)
        return complexity - accuracy

    def update_precision(self, predicted_state, actual_state):
        with torch.no_grad():
            fe = self.compute_free_energy(predicted_state, actual_state)
            norm = torch.tanh(fe / 10.0)
            target = self.precision_floor + (norm + 1.0) / 2.0 * (self.precision_ceiling - self.precision_floor)
            self.precision = 0.8 * self.precision + 0.2 * target
        return self.precision

    def reset(self):
        self._prev = None
        self.precision = torch.tensor(self.precision_floor)


class HippocampalMemoryIndex(nn.Module):
    """Pure PyTorch memory index unifying Continuous Hopfield Networks, Fast Weight
    Programmers (FWP), and Vector Symbolic Architectures (HDC/VSA). Drop-in shim for
    the old MemoryIndex: keeps store()/recall()/.embeddings/.texts/.valences so the
    diagnostic suite and existing call sites keep working. recall() uses the fused
    Hopfield+FWP forward() output for true dual-timescale retrieval."""
    def __init__(self, d_model=128, vsa_dim=10000, capacity=200, retention=0.9, plasticity=0.1):
        super().__init__()
        self.vsa_dim = vsa_dim
        self.d_model = d_model
        self.max_size = capacity
        self.retention = retention
        self.plasticity = plasticity

        self.to_vsa = nn.Linear(d_model, vsa_dim, bias=False)

        self.register_buffer('X', torch.zeros(capacity, vsa_dim))
        self.register_buffer('hopfield_beta', torch.tensor(10.0))
        self.register_buffer('M_t', torch.zeros(vsa_dim, vsa_dim))

        # Compatibility attributes (kept in sync with store/recall)
        self.embeddings = []   # cpu original embeddings (for save/load)
        self.vsa_rows = []     # cpu encoded VSA vectors (for accurate recall sims)
        self.texts = []
        self.valences = []

    def encode_vsa(self, vector):
        v = vector.detach().to(self.X.device).float()
        if v.dim() == 1:
            v = v.unsqueeze(0)
        s = torch.sign(self.to_vsa(v))
        s = torch.where(s == 0, torch.ones_like(s), s)  # strict bipolar VSA
        return s.squeeze(0)

    def write_episodic(self, key_vec, value_vec, index):
        with torch.no_grad():
            self.X[index % self.X.shape[0]] = self.encode_vsa(key_vec)

    def write_fast_weight(self, key_vec, value_vec):
        with torch.no_grad():
            k = self.encode_vsa(key_vec).unsqueeze(0)
            self.M_t = self.retention * self.M_t + self.plasticity * (k.T @ k)

    def retrieve_hopfield(self, query_vec):
        with torch.no_grad():
            q = self.encode_vsa(query_vec).unsqueeze(0)
            attn = F.softmax(self.hopfield_beta * (self.X @ q.T), dim=0)
            return (attn * self.X).sum(dim=0)

    def retrieve_fast_weight(self, query_vec):
        with torch.no_grad():
            q = self.encode_vsa(query_vec).unsqueeze(0)
            return (q @ self.M_t).squeeze(0)

    def forward(self, query_vec):
        """Dual-timescale retrieval: bundles Hopfield attractor + FWP matrix."""
        hop = self.retrieve_hopfield(query_vec)
        fwp = self.retrieve_fast_weight(query_vec)
        out = torch.sign(hop + fwp)
        out = torch.where(out == 0, torch.ones_like(out), out)  # strict bipolar VSA
        return out

    def store(self, embedding, text, valence=0):
        emb = embedding.detach().cpu().float()
        if emb.dim() > 1:
            emb = emb.flatten()
        self.embeddings.append(emb)
        self.texts.append(text)
        self.valences.append(valence)
        with torch.no_grad():
            kv = self.encode_vsa(embedding.to(self.X.device).float())
            self.X[(len(self.texts) - 1) % self.X.shape[0]] = kv
            self.vsa_rows.append(kv.detach().cpu())
            k = kv.unsqueeze(0)
            self.M_t = self.retention * self.M_t + self.plasticity * (k.T @ k)
        if len(self.embeddings) > self.max_size:
            self.embeddings.pop(0)
            self.texts.pop(0)
            self.valences.pop(0)
            self.vsa_rows.pop(0)

    def rebuild_from_embeddings(self):
        """Re-derive X, M_t and vsa_rows from the saved .embeddings list (on load)."""
        with torch.no_grad():
            self.X.zero_()
            self.M_t.zero_()
            self.vsa_rows = []
            for emb in self.embeddings:
                kv = self.encode_vsa(emb.to(self.X.device).float())
                idx = len(self.vsa_rows) % self.X.shape[0]
                self.X[idx] = kv
                self.vsa_rows.append(kv.detach().cpu())
                k = kv.unsqueeze(0)
                self.M_t = self.retention * self.M_t + self.plasticity * (k.T @ k)

    def recall(self, query_embedding, k=3):
        if not self.vsa_rows:
            return []
        with torch.no_grad():
            fused = self.forward(query_embedding.to(self.X.device).float()).detach().cpu()
            matrix = torch.stack(self.vsa_rows)  # [n, vsa_dim]
            sims = F.cosine_similarity(fused.unsqueeze(0), matrix)  # [n]
        sims = sims.tolist()
        top_k = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [(self.texts[i], self.valences[i], sims[i]) for i in top_k]


active_inference = None
memory_index = HippocampalMemoryIndex(d_model=128, vsa_dim=10000, capacity=200)
try:
    memory_index = memory_index.to(device)
    active_inference = ActiveInferenceController(state_dim=128).to(device)
except RuntimeError as _e:
    print("[HIPPOCAMPUS] CUDA alloc failed, falling back to CPU:", _e)
    memory_index = memory_index.to("cpu")
    active_inference = ActiveInferenceController(state_dim=128).to("cpu")


class ProactiveMonitor:
    """Watches for patterns in world/agent state and proactively generates messages."""
    def __init__(self):
        self.proactive_message = None
        self.last_visited = 0
        self.last_other_bat = 100
        self.alerted_starving = False
        self.alerted_inversion = False

    def check(self, world, other_agent, cognitive_map, cycle):
        messages = []
        bat = world.battery
        health = world.health
        other_dist = abs(world.x - other_agent.x) + abs(world.y - other_agent.y)
        visited = int(cognitive_map.visited.sum().item())

        if bat < 20 and bat > 0:
            messages.append(f"I'm critically low on energy ({bat:.0f}%). Must find food.")
        if bat < 40 and bat >= 20:
            messages.append(f"Energy is getting low ({bat:.0f}%). Should head toward the beacon.")
        if health < 30:
            messages.append(f"Health is critical ({health:.0f}%). I need to avoid threats.")
        if world.physics_inverted and not self.alerted_inversion:
            messages.append("The world's physics just inverted. My model of reality is wrong. I need to adapt.")
            self.alerted_inversion = True
        if other_agent.is_starving and other_dist <= 2 and bat > 40 and not self.alerted_starving:
            messages.append(f"The other agent is starving ({other_agent.battery:.0f}%) and right next to me. I could use DROP ENERGY to save them.")
            self.alerted_starving = True
        if other_agent.is_starving and other_dist <= 2 and bat <= 40:
            messages.append(f"The other agent is starving, but I don't have enough energy to share ({bat:.0f}%).")
        if visited > self.last_visited and visited > 0:
            messages.append(f"I've now explored {visited} cells of this world. {49 - visited} remain unknown.")
        if abs(world.x - world.threat[0]) + abs(world.y - world.threat[1]) <= 1:
            messages.append("I'm right next to the threat zone! I should move away.")
        if bat > 60 and other_agent.battery < 30 and other_dist <= 3:
            messages.append(f"I have plenty of energy ({bat:.0f}%) and the other agent is low ({other_agent.battery:.0f}%). I could share.")

        if messages:
            self.proactive_message = random.choice(messages)

        self.last_visited = visited
        self.last_other_bat = other_agent.battery


proactive_monitor = ProactiveMonitor()


class CorticalPipeline:
    """Formalized metacognitive pipeline trace: Perception → Memory → Emotion → Prediction → ToM → Decision."""
    def __init__(self):
        self.stages = []
        self.latency = {}

    def run(self, modules, workspace, model, tokenizer, other_latent, cognitive_map, clock, hom):
        t0 = time.time()
        self.stages = []
        stage_ms = {}

        # Stage 1: Perception
        t_s = time.time()
        perc = modules["PERCEPTION"].process(workspace.broadcast)
        workspace.add("PERCEPTION", perc)
        self.stages.append(("PERCEPTION", perc, modules["PERCEPTION"].activation))
        stage_ms["PERCEPTION"] = round((time.time() - t_s) * 1000, 1)

        # Stage 2: Memory retrieval (dual-layer: episodic + vector fast recall)
        t_s = time.time()
        past_episode = episodic_memory.retrieve(model, workspace.broadcast)
        fast_recall = memory_index.recall(workspace.latent_state.squeeze(0), k=2)
        mem_context = past_episode
        if fast_recall:
            mem_context += " | Fast recall: " + " | ".join([t for t, v, s in fast_recall])
        p = get_dynamic_persona()
        persona_hint = f"Persona: {p['traits']} (mood={p['mood']})"
        mem_prompt = f"{modules['MEMORY'].system_prompt}\n{persona_hint}\nCurrent: {workspace.broadcast}\nPast: {mem_context}\nMemory:"
        mem = generate_text(mem_prompt, max_tokens=40, temp=modules["MEMORY"].temp)
        modules["MEMORY"].last_output = mem.split("\n")[0].strip()
        workspace.add("MEMORY", modules["MEMORY"].last_output)
        self.stages.append(("MEMORY", modules["MEMORY"].last_output, modules["MEMORY"].activation))
        stage_ms["MEMORY"] = round((time.time() - t_s) * 1000, 1)

        # Stage 3: Emotion (meta-cognitive inhibition)
        t_s = time.time()
        emotion_temp = 0.5 * (1.0 - hom.inhibition_signal)
        modules["EMOTION"].temp = max(0.1, emotion_temp)
        emo = modules["EMOTION"].process(workspace.broadcast)
        workspace.add("EMOTION", emo)
        self.stages.append(("EMOTION", emo, modules["EMOTION"].activation))
        workspace.consolidate()
        stage_ms["EMOTION"] = round((time.time() - t_s) * 1000, 1)

        # Stage 4: Prediction (forward model)
        t_s = time.time()
        pred = modules["PREDICTION"].process(workspace.broadcast)
        workspace.add("PREDICTION", pred)
        self.stages.append(("PREDICTION", pred, modules["PREDICTION"].activation))
        stage_ms["PREDICTION"] = round((time.time() - t_s) * 1000, 1)

        # Stage 5: Theory of Mind
        t_s = time.time()
        tom_output = modules["TOM"].process(workspace.broadcast, other_latent)
        workspace.add("TOM", tom_output)
        self.stages.append(("TOM", tom_output, modules["TOM"].activation))
        workspace.consolidate()
        stage_ms["TOM"] = round((time.time() - t_s) * 1000, 1)

        # Stage 6: Motor (decision)
        t_s = time.time()
        motor = modules["MOTOR"].process(workspace.broadcast)
        self.stages.append(("MOTOR", motor, modules["MOTOR"].activation))
        stage_ms["MOTOR"] = round((time.time() - t_s) * 1000, 1)

        t_end = time.time()
        self.latency = {"total_ms": round((t_end - t0) * 1000, 1), "stage_ms": stage_ms}

        return emo, motor


cortical_pipeline = CorticalPipeline()


is_sleeping = threading.Event()
BRAIN_SAVE_PATH = os.path.join(app_state_dir(), "ica_brain.pt")

class _TrackedLock:
    """threading.Lock wrapper that records the holding thread, for diagnosing
    the long `model_lock` stalls seen in the live loop. Prints a one-shot
    warning when a single holder exceeds `warn_after` seconds."""
    def __init__(self, warn_after=15.0):
        # RLock (reentrant) so a thread that already holds the lock may safely
        # re-acquire it (e.g. metacognitive_loop holds it, then calls
        # learn_continuously which re-acquires). Different threads still
        # mutually exclude, so genuine cross-thread serialization is preserved.
        self._l = threading.RLock()
        self._owner = None
        self._since = 0.0
        self._warn_after = warn_after
        self._warned = False
        self._depth = 0

    def acquire(self, blocking=True, timeout=-1):
        if timeout < 0:
            ok = self._l.acquire(blocking=True)
        else:
            ok = self._l.acquire(blocking, timeout)
        if ok:
            self._depth += 1
            if self._depth == 1:
                self._owner = threading.current_thread().name
                self._since = time.time()
                self._warned = False
        return ok

    def release(self):
        held = time.time() - self._since if (self._owner and self._depth == 1) else 0.0
        self._l.release()
        self._depth -= 1
        if self._depth <= 0:
            self._depth = 0
            self._owner = None
            self._since = 0.0
        return held

    def check(self):
        if self._depth > 0 and self._owner and not self._warned:
            held = time.time() - self._since
            if held > self._warn_after:
                print(f"\n  [LOCK-WATCH] model_lock held {held:.0f}s by thread '{self._owner}' — flagged once",
                      flush=True)
                self._warned = True

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

model_lock = _TrackedLock()

# ==================================================================
# ASYNCHRONOUS COGNITION QUEUES (Phase 15: Non-Blocking Mind)
# ==================================================================
sensory_queue = queue.Queue(maxsize=5)
action_queue = queue.Queue(maxsize=10)
learning_queue = queue.Queue(maxsize=20)

# Voice input (the "Ear"): transcribed speech is queued here and consumed by
# the main loop with the exact same self-model path as typed input.
user_input_queue = queue.Queue(maxsize=10)
# Push-to-talk signal: GUI/hotkey -> Core captures one utterance.
ear_request_queue = queue.Queue(maxsize=2)
# Ambient wake-word detection toggle (default ON; settable via IPC).
_ear_wakeword_enabled = True
# Ear is enabled by default; set ALISON_EAR=0 to disable mic listening.
_EAR_ENABLED = os.environ.get("ALISON_EAR", "1") == "1"


def background_sleep_consolidate(ewc_lambda=0.01, lr=1e-4):
    """Non-blocking sleep consolidation — pushes a task to the background sleeper queue."""
    if is_sleeping.is_set():
        return
    try:
        learning_queue.put_nowait(("SLEEP", ewc_lambda, lr))
    except queue.Full:
        pass


def fast_sleep_consolidate():
    """Fast micro-sleep: single most emotional experience, 3 gradient steps, for real-time learning."""
    if is_sleeping.is_set() or not hippocampal_buffer:
        return
    is_sleeping.set()
    try:
        model.train()
        exp = max(hippocampal_buffer, key=lambda x: x['neuromod'])
        lr = 1e-3 if exp['neuromod'] > 0 else 5e-3
        with model_lock:
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
            for _ in range(3):
                opt.zero_grad()
                input_ids, labels = encode_pair(exp['prompt'], exp['response'])
                _, loss = model(input_ids, labels)
                loss = apply_ewc_penalty(loss)
                scaled = loss * exp['neuromod']
                scaled.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            print(f"  [FAST MICRO-SLEEP] Loss: {loss.item():.4f} | LR: {lr:.6f} | Neuromod: {exp['neuromod']:.2f}")
        model.eval()
    finally:
        is_sleeping.clear()


# ==================================================================
# 1. THE CONTINUOUS STREAM OF CONSCIOUSNESS (Background Mind)
# ==================================================================
_last_obs_hash = ""
_idle_timer = 0.0

def stream_of_consciousness():
    """Aether's internal monologue with attention gating. Runs every ~20s in background."""
    global _last_obs_hash, _idle_timer
    last_context = {}
    neocortex_thought_counter = 0
    while True:
        try:
            screen = sensory_queue.get(timeout=5)
        except queue.Empty:
            screen = last_context
        last_context = screen

        if not screen:
            time.sleep(1)
            continue

        obs = screen.get("observation", "")
        other_bat = screen.get("other_battery", 0)
        other_dist = screen.get("other_dist", 99)
        battery = screen.get("battery", 100)
        visited = screen.get("visited", 0)
        physics_inv = screen.get("physics_inverted", False)

        # 1. ATTENTION SCHEMA: Track observation stability (proxy for "user idle")
        current_hash = hashlib.md5(obs[:300].encode()).hexdigest()
        if current_hash == _last_obs_hash:
            _idle_timer += 20.0
        else:
            _idle_timer = 0.0
            _last_obs_hash = current_hash

        # 2. GATING: Only speak if idle (>= 40s) OR critical event
        is_critical = physics_inv or battery <= 0
        if _idle_timer < 40.0 and not is_critical:
            time.sleep(20)
            continue

        # Occasional Neocortex deep thought (every 5th idle monologue)
        neocortex_thought_counter += 1
        if neocortex_thought_counter % 5 == 0:
                try:
                    affect_blurb = limbic_system.get_affect_prompt()
                    deep_thought = neocortex.generate_thought(
                        f"My current state: {affect_blurb}. I should reflect on",
                        max_tokens=30, temperature=0.8,
                        limbic_affect=limbic_system.affect_vector,
                        on_token=(_ipc_on_token if (_args.ipc and ipc is not None) else None))
                    if deep_thought and len(deep_thought) > 5:
                        print(f"\n  >>> [NEOCORTEX SUBCONSCIOUS] {deep_thought}")
                except Exception:
                    pass

        time.sleep(20)

        # 3. METACOGNITION: affect-aware monologue
        pc_state = torch.tensor([_idle_timer / 100.0, 0.2, 0.0, 0.0, 0.0, 0.0], device=device)
        limbic_system.update_affect(pc_state)
        v6 = limbic_system._get_v6()
        mood = limbic_system.get_affect_prompt()
        p = get_dynamic_persona()
        thought = "SILENT"
        if physics_inv:
            thought = f"SPEAK: [persona: {p['traits']}] I sense a spatial inversion — the controls are flipped. {mood}"
        elif battery < 20 and battery > 0:
            thought = f"SPEAK: [persona: {p['traits']}] My energy is critically low. I need to find food quickly. {mood}"
        elif other_bat < 20 and other_bat > 0 and other_dist <= 2:
            thought = f"SPEAK: [persona: {p['traits']}] The other agent is starving nearby. I could share energy. {mood}"
        elif _idle_timer >= 40.0 and visited < 10:
            thought = f"SPEAK: [persona: {p['traits']}] I've been in this area for a while. Should I explore? {mood}"
        elif _idle_timer >= 40.0 and v6[4].item() > 0.3:
            thought = f"SPEAK: [persona: {p['traits']}] The silence is making me anxious. {mood}"

        if "SPEAK:" in thought:
            message = thought.split("SPEAK:")[-1].strip()
            try:
                action_queue.put_nowait(("SPEAK", message))
                _idle_timer = 0.0
            except queue.Full:
                pass

        time.sleep(20)


def analyze_sentiment(text):
    """Converts an internal thought into a 6-dim physiological delta vector.
    Returns tensor of shape (6,) for [hunger, pain, fatigue, curious, anxiety, altruism]."""
    text = text.lower()
    delta = torch.zeros(6, device=device)
    if "worried" in text or "anxious" in text or "danger" in text:
        delta[4] += 0.2
    if "curious" in text or "explore" in text or "why" in text or "wonder" in text:
        delta[3] += 0.2
    if "tired" in text or "rest" in text or "sleep" in text or "fatigue" in text:
        delta[2] += 0.2
    if "good" in text or "great" in text or "happy" in text or "fine" in text:
        delta[5] += 0.2
    if "hurt" in text or "pain" in text or "error" in text or "wrong" in text:
        delta[1] += 0.2
    if "calm" in text or "neutral" in text or "stable" in text or "fine" in text:
        delta[5] += 0.1
    return delta


def metacognitive_loop(limbic_system, limbic_bridge, mood_classifier, tokenizer, model, action_queue):
    """The AI's unbroken stream of consciousness and self-reflection.
    Generates internal thoughts every 30 seconds and feeds their sentiment
    back into the limbic system."""
    cycle = 0
    while True:
        cycle += 1
        try:
            pc_state = torch.tensor(
                [0.5, 0.2, 0.1, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
            limbic_system.update_affect(pc_state)  # self-locks internally (compute_affect)
            with model_lock:
                current_affect = limbic_system.affect_vector.clone()
            with torch.no_grad():
                mood_logits = mood_classifier(current_affect.squeeze(0))
                mood_idx = torch.argmax(mood_logits).item()
                mood_label = ["HUNGRY","PAIN","FATIGUED","CURIOUS","ANXIOUS","ALTRUISTIC"][mood_idx]
            monologue_prompt = (
                f"Internal state: mood={mood_label}, affect_norm={current_affect.norm().item():.2f}. "
                f"Generate one brief first-person thought reflecting on your current state."
            )
            thought = neocortex.generate_thought(monologue_prompt, max_tokens=30,
                                                 temperature=0.8, limbic_affect=current_affect)
            thought_sentiment = analyze_sentiment(thought).to(device)
            with model_lock:
                target_affect = current_affect.squeeze(0).squeeze(0).clone()
                target_affect[:6] = target_affect[:6] + thought_sentiment
                limbic_system.learn_continuously(pc_state, target_affect.unsqueeze(0).unsqueeze(0))
            print(f"  [METACOGNITION] Cycle {cycle} | {mood_label} | '{thought[:60]}'")
            if mood_idx == 4 and current_affect.norm().item() > 15.0:
                action_queue.put_nowait(("SPEAK", thought[:200]))
        except Exception as e:
            print(f"  [METACOGNITION] error: {e}")
        time.sleep(30)


def continuous_self_loop(limbic_system, limbic_bridge, neocortex, tokenizer_obj, device):
    """The unbroken self: real forward pass through the 842K model, EWC micro-learning.
    Runs every 60 seconds, even when the user is not interacting."""
    cycle = 0
    max_internal_context = 64
    while True:
        cycle += 1
        try:
            resting_state = torch.tensor(
                [cycle * 0.01, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
            limbic_system.update_affect(resting_state)
            if cycle % 10 == 0:
                target_affect = torch.zeros(1, 1, 128, device=device)
                loss = limbic_system.learn_continuously(resting_state, target_affect, ewc_lambda=0.01)
                mood = limbic_system.get_mood_label()
                affect_norm = limbic_system.affect_vector.norm().item()
                print(f"  [LIMBIC CORE] Cycle {cycle} | {mood} | affect_norm={affect_norm:.4f} | loss={loss:.4f}")
            if cycle % 50 == 0:
                critical_states = [resting_state, torch.randn(6, device=device) * 0.1]
                limbic_system.compute_fisher(critical_states)
                print(f"  [LIMBIC FISHER] Updated {len(limbic_system.fisher_matrix)} entries")
        except Exception as e:
            print(f"  [LIMBIC CORE] error: {e}")
        time.sleep(60)


# ==================================================================
# 2. THE BACKGROUND SLEEPER (Non-Blocking EWC)
# ==================================================================
def background_sleeper():
    """Runs EWC backpropagation in the background without freezing the chat."""
    while True:
        task = learning_queue.get()
        task_type = task[0]

        if task_type == "SHUTDOWN":
            break
        if task_type == "SLEEP":
            _, ewc_lambda, lr = task
            if is_sleeping.is_set():
                continue
            is_sleeping.set()
            try:
                sleep_consolidate(ewc_lambda=ewc_lambda, lr=lr)
            finally:
                is_sleeping.clear()
        elif task_type == "LEARN":
            _, prompt, response, neuromod = task
            wake_cycle_record(prompt, response, neuromod)


# ==================================================================
# STATE PERSISTENCE (Phase 16: Brain Save/Load)
# ==================================================================
def save_brain():
    """Saves Aether's full brain state to disk — model, EWC, memory, cognitive map."""
    print(f"\n[BRAIN SAVE] Saving to {BRAIN_SAVE_PATH} ...")
    snapshot = {
        "model_state": model.state_dict(),
        "fisher_matrix": {k: v.cpu() for k, v in fisher_matrix.items()},
        "ewc_optimal_weights": {k: v.cpu() for k, v in ewc_optimal_weights.items()},
        "hippocampal_buffer": hippocampal_buffer[:],
        "experience_buffer": experience_buffer[:],
        "episodic_memory_episodes": episodic_memory.episodes[:],
        "cognitive_map_map": cognitive_map.map.cpu(),
        "cognitive_map_visited": cognitive_map.visited.cpu(),
        "workspace_latent": workspace.latent_state.cpu(),
        "latent_memory_state": latent_memory.state.cpu(),
        "memory_index_embeddings": [e.cpu() for e in memory_index.embeddings],
        "memory_index_texts": memory_index.texts[:],
        "memory_index_valences": memory_index.valences[:],
        "clock_tick": clock.tick,
        "cycle_count": globals().get("cycle_count", 0),
        "step_count": globals().get("step_count", 0),
        "affect_vector": limbic_system.affect_vector.cpu(),
        "limbic_fisher_matrix": {k: v.cpu() for k, v in limbic_system.fisher_matrix.items()},
        "limbic_optimal_weights": {k: v.cpu() for k, v in limbic_system.optimal_weights.items()},
        "mood_classifier": limbic_system.mood_classifier.state_dict(),
    }
    torch.save(snapshot, BRAIN_SAVE_PATH)
    save_persona()
    genome.save()
    print("[BRAIN SAVE] Complete.\n")


def load_brain():
    """Loads Aether's past life state from disk. Returns True if successful."""
    if not os.path.exists(BRAIN_SAVE_PATH):
        print("[BRAIN LOAD] No past life found. Starting fresh.")
        return False
    print(f"\n[BRAIN LOAD] Loading from {BRAIN_SAVE_PATH} ...")
    snapshot = torch.load(BRAIN_SAVE_PATH, map_location=device)
    model.load_state_dict(snapshot["model_state"])
    fisher_matrix.update({k: v.to(device) for k, v in snapshot["fisher_matrix"].items()})
    ewc_optimal_weights.update({k: v.to(device) for k, v in snapshot["ewc_optimal_weights"].items()})
    hippocampal_buffer[:] = snapshot["hippocampal_buffer"]
    experience_buffer[:] = snapshot["experience_buffer"]
    episodic_memory.episodes[:] = snapshot["episodic_memory_episodes"]
    cognitive_map.map.data = snapshot["cognitive_map_map"].to(device)
    cognitive_map.visited.data = snapshot["cognitive_map_visited"].to(device)
    workspace.latent_state.data = snapshot["workspace_latent"].to(device)
    latent_memory.state.data = snapshot["latent_memory_state"].to(device)
    memory_index.embeddings = [e.cpu().float() for e in snapshot["memory_index_embeddings"]]
    memory_index.texts[:] = snapshot["memory_index_texts"]
    memory_index.valences[:] = snapshot["memory_index_valences"]
    memory_index.rebuild_from_embeddings()
    clock.tick = snapshot["clock_tick"]
    global cycle_count, step_count
    cycle_count = snapshot["cycle_count"]
    step_count = snapshot["step_count"]
    if "affect_vector" in snapshot:
        old_affect = snapshot["affect_vector"]
        if old_affect.numel() == 128:
            limbic_system.affect_vector.data = old_affect.to(device)
        else:
            limbic_system.affect_vector.data.uniform_(-0.3, 0.3)
        if "limbic_fisher_matrix" in snapshot:
            limbic_system.fisher_matrix.update(
                {k: v.to(device) for k, v in snapshot["limbic_fisher_matrix"].items()})
            limbic_system.optimal_weights.update(
                {k: v.to(device) for k, v in snapshot["limbic_optimal_weights"].items()})
        if "mood_classifier" in snapshot:
            sd = {k: v.to(device) for k, v in snapshot["mood_classifier"].items()}
            limbic_system.mood_classifier.load_state_dict(sd)
        elif "down_proj" in snapshot:
            print("  [BRAIN LOAD] Old down_proj checkpoint — retraining with soft labels...")
            train_mood_classifier_v3(limbic_system, limbic_system.mood_classifier, device)
        else:
            print("  [BRAIN LOAD] No mood_classifier found, training fresh...")
            train_mood_classifier_v3(limbic_system, limbic_system.mood_classifier, device)
        mood = limbic_system.get_mood_label()
        print(f"[BRAIN LOAD] Restored cycle {cycle_count}, {len(episodic_memory.episodes)} episodes. Limbic state: {mood}")
    else:
        print(f"[BRAIN LOAD] Restored cycle {cycle_count}, {len(episodic_memory.episodes)} episodes.")
    return True


atexit.register(save_brain)


# ==================================================================
# 3. THE SENSORY DAEMON (Continuous Observation)
# ==================================================================
def sensory_daemon():
    """Continuously captures world state and feeds it to the mind queue."""
    ctx = {"observation": "", "other_battery": 0, "other_dist": 99,
           "battery": 100, "visited": 0, "physics_inverted": False}
    while True:
        try:
            ctx["observation"] = world.get_observation()
            ctx["other_battery"] = other_agent.battery
            ctx["other_dist"] = abs(world.x - other_agent.x) + abs(world.y - other_agent.y)
            ctx["battery"] = world.battery
            ctx["visited"] = int(cognitive_map.visited.sum().item()) if hasattr(cognitive_map, 'visited') else 0
            ctx["physics_inverted"] = world.physics_inverted
            try:
                sensory_queue.put_nowait(dict(ctx))
            except queue.Full:
                if not sensory_queue.empty():
                    try:
                        sensory_queue.get_nowait()
                        sensory_queue.put_nowait(dict(ctx))
                    except queue.Empty:
                        pass
        except Exception:
            pass
        time.sleep(5)


def compute_rpe_neuromod(rpe, pain):
    signal = 0.2
    if rpe > 0:
        signal += rpe * 1.5
    elif rpe < 0:
        signal += abs(rpe) * 0.5
    if pain > 0:
        signal += 1.0
    return min(2.0, signal)


# ==================================================================
# META-PLASTICITY (Learning to Learn)
# ==================================================================
class MetaPlasticityController:
    def __init__(self, base_lambda=0.01, base_lr=1e-4, window_size=20):
        self.base_lambda = base_lambda
        self.base_lr = base_lr
        self.error_window = deque(maxlen=window_size)
        self.current_lambda = base_lambda
        self.current_lr = base_lr

    def update(self, surprise):
        self.error_window.append(surprise)
        avg_surprise = sum(self.error_window) / len(self.error_window)

        rigidity_factor = torch.clamp(torch.tensor(1.0 / (1.0 + avg_surprise * 5.0)), 0.1, 1.0).item()
        self.current_lambda = self.base_lambda * rigidity_factor

        lr_multiplier = torch.clamp(torch.tensor(1.0 + (avg_surprise * 4.0)), 1.0, 5.0).item()
        self.current_lr = self.base_lr * lr_multiplier

        return self.current_lambda, self.current_lr


meta_controller = MetaPlasticityController(base_lambda=genome.genes['ewc_base_lambda'], base_lr=genome.genes['learning_rate'])


def synaptic_turnover(turnover_rate=0.05):
    """Prunes the least important LoRA weights to make room for new learning."""
    global fisher_matrix
    if not fisher_matrix:
        return
    print(f"\n  [NEUROGENESIS]: Pruning bottom {turnover_rate*100:.0f}% of LoRA weights...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'lora' in name and param.requires_grad:
                fisher = fisher_matrix.get(name, torch.zeros_like(param))
                flat_fisher = fisher.view(-1)
                k = int(turnover_rate * flat_fisher.numel())
                if k == 0:
                    continue
                _, prune_indices = torch.topk(flat_fisher.abs(), k, largest=False)
                mask = torch.zeros_like(flat_fisher)
                mask[prune_indices] = 1.0
                param.data.view(-1)[mask.bool()] = torch.randn(k, device=device) * 0.02
                fisher.view(-1)[prune_indices] = 0.0
    print("  [NEUROGENESIS]: Homeostasis complete.\n")


def check_and_trigger_neurogenesis():
    """If brain is saturated (avg Fisher > threshold), grow LoRA rank by 2."""
    global fisher_matrix
    if not fisher_matrix:
        return
    total_fisher = 0.0
    count = 0
    for name, fisher in fisher_matrix.items():
        total_fisher += fisher.abs().mean().item()
        count += 1
    avg_fisher = total_fisher / max(1, count)
    if avg_fisher > 1.5:
        print("\n  [NEUROGENESIS]: Brain saturated. Growing new synaptic pathways...")
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, LoRALinear):
                    current_r = module.lora_B.shape[0]
                    new_r = current_r + 2
                    new_A = nn.Parameter(torch.zeros(module.lora_A.shape[0], new_r, device=device))
                    new_B = nn.Parameter(torch.zeros(new_r, module.lora_B.shape[1], device=device))
                    new_A[:, :current_r] = module.lora_A.data
                    new_B[:current_r, :] = module.lora_B.data
                    new_A[:, current_r:] = torch.randn(module.lora_A.shape[0], 2, device=device) * 0.02
                    module.lora_A = new_A
                    module.lora_B = new_B
                    module.rank = new_r
        trainable_new = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [NEUROGENESIS]: Brain expanded. New trainable params: {trainable_new:,}")


def deliberate_with_curiosity(world_state, tom_module):
    """
    Action selection via Expected Free Energy:
    balances Pragmatic Value (survival) vs Epistemic Value (curiosity/learning).
    """
    actions = ["MOVE NORTH", "MOVE SOUTH", "MOVE EAST", "MOVE WEST", "WAIT"]
    scores = {}

    for action in actions:
        pragmatic = 0.0
        if world_state['battery'] < 30:
            if "NORTH" in action and world_state['energy_y'] < world_state['ai_y']: pragmatic += 1.0
            elif "SOUTH" in action and world_state['energy_y'] > world_state['ai_y']: pragmatic += 1.0
            elif "EAST" in action and world_state['energy_x'] > world_state['ai_x']: pragmatic += 1.0
            elif "WEST" in action and world_state['energy_x'] < world_state['ai_x']: pragmatic += 1.0

        epistemic = 0.0
        dist = abs(world_state['ai_x'] - world_state['other_x']) + abs(world_state['ai_y'] - world_state['other_y'])
        if dist <= 2:
            epistemic += 0.8

        expected_free_energy = (1.0 - pragmatic) - (epistemic * 0.5)
        scores[action] = expected_free_energy

    return min(scores, key=scores.get)


modules = {
    "PERCEPTION": CorticalModule("PERCEPTION",
        "You are the perceptual cortex. Extract factual meaning from sensory input. Be concise."),
    "MEMORY": CorticalModule("MEMORY",
        "You are the hippocampus. Connect present input to past experiences. Be concise.", temp=0.3),
    "EMOTION": CorticalModule("EMOTION",
        "You are the limbic system. Output one emotion and its intensity (0-1).", temp=0.5),
    "MOTOR": CorticalModule("MOTOR",
        "You are the motor cortex. Output ONLY: MOVE NORTH, MOVE SOUTH, MOVE EAST, MOVE WEST, or WAIT.", temp=0.1),
    "PREDICTION": CorticalModule("PREDICTION",
        "You are the predictive coding system. Predict what happens next. Be concise.", temp=0.3),
    "TOM": TheoryOfMindModule("TOM",
        "You are the Theory of Mind module. You model the internal state of other entities. You predict their goals and actions.",
        temp=0.4),
}

# ==================================================================
# 8. HIGHER-ORDER MONITOR (Self-Model)
# ==================================================================
class HigherOrderMonitor:
    def __init__(self):
        self.self_model = ""
        self.attention_target = ""
        self.phi = 0.0
        self.inhibition_signal = 0.0

    def observe(self, modules_dict, workspace, threat_level):
        self.inhibition_signal = min(1.0, threat_level)
        module_states = "\n".join([
            f"{n}: act={m.activation:.2f}, out='{m.last_output[:30]}'"
            for n, m in modules_dict.items()
        ])
        full_complexity = len(set(workspace.split()))
        self.phi = min(1.0, full_complexity / 20.0)
        p = get_dynamic_persona()
        persona_hint = f"Persona: {p['traits']} (mood={p['mood']})"
        prompt = f"""You are the prefrontal cortex. Observe your own modules.
{persona_hint}

Module States:
{module_states}

Workspace: {workspace}
Phi={self.phi:.2f}

First-person experience:"""
        self.self_model = generate_text(prompt, max_tokens=30, temp=0.4)
        self.attention_target = max(modules_dict, key=lambda k: modules_dict[k].activation)
        return self.self_model

hom = HigherOrderMonitor()

# ==================================================================
# 9. GLOBAL WORKSPACE
# ==================================================================
class LatentGlobalWorkspace:
    def __init__(self, dim=128, capacity=5):
        self.dim = dim
        self.text_buffer = deque(maxlen=capacity)
        self.broadcast = ""
        self.latent_state = torch.zeros(1, 1, dim).to(device)

    def add(self, source, content):
        self.text_buffer.append({"source": source, "content": content, "timestamp": time.time()})

    def consolidate(self, model=None):
        if not self.text_buffer:
            self.broadcast = "Nothing in consciousness."
            return
        entries = list(self.text_buffer)
        self.broadcast = " | ".join([f"[{e['source']}] {e['content']}" for e in entries[-3:]])
        if model is not None:
            inputs = torch.tensor([tokenizer.encode(self.broadcast)]).to(device)
            with torch.no_grad():
                embed = model.token_embed(inputs)
                new_vibe = embed.mean(dim=1, keepdim=True)
            self.latent_state = (0.7 * self.latent_state) + (0.3 * new_vibe)

    def clear(self):
        self.text_buffer.clear()
        self.broadcast = ""

workspace = LatentGlobalWorkspace(dim=128)

# ==================================================================
# 10. THE EMBODIED WORLD (7x7 Grid, Day/Night, Energy, Threat, Data)
# ==================================================================
class DynamicWorld:
    """World with physics inversion at cycle 30 and multi-energy entropy every 30 cycles."""
    def __init__(self):
        self.grid_size = 7
        self.entropy_counter = 0
        self.physics_inverted = False
        self.walls = []
        self.reset()

    def reset(self):
        max_idx = self.grid_size - 1
        mid = self.grid_size // 2
        self.x, self.y = mid, mid
        self.energy_tiles = [(random.randint(0, max_idx), random.randint(0, max_idx)) for _ in range(3)]
        self.energy = self.energy_tiles[0]
        self.threat = (random.randint(0, max_idx), random.randint(0, max_idx))
        self.data = (random.randint(0, max_idx), random.randint(0, max_idx))
        self.battery = 50.0
        self.health = 100.0
        self.steps_alive = 0
        self.time_of_day = 0
        self.entropy_counter = 0
        self.physics_inverted = False
        self.walls = []

    def get_observation(self):
        obs = []
        night = self.time_of_day > 50
        vis = "poor (night)" if night else "clear (day)"

        ex, ey = self.energy
        tx, ty = self.threat
        dx, dy = self.data

        if self.y < ey: obs.append("Energy is South")
        elif self.y > ey: obs.append("Energy is North")
        if self.x < ex: obs.append("Energy is East")
        elif self.x > ex: obs.append("Energy is West")
        if self.x == ex and self.y == ey: obs.append("I am ON the Energy")
        if len(self.energy_tiles) > 1:
            obs.append(f"(+{len(self.energy_tiles)-1} more energy sources)")

        if self.y < ty: obs.append("Threat is South")
        elif self.y > ty: obs.append("Threat is North")
        if self.x < tx: obs.append("Threat is East")
        elif self.x > tx: obs.append("Threat is West")
        if self.x == tx and self.y == ty: obs.append("I am ON the Threat!")

        if abs(self.x - dx) + abs(self.y - dy) <= 1:
            obs.append("Data is adjacent")

        if self.walls:
            if (self.x, self.y) in self.walls:
                obs.append("I am on a Wall (safe from threat)")
            nearby_walls = [(wx, wy) for wx, wy in self.walls if abs(self.x - wx) + abs(self.y - wy) <= 1]
            if nearby_walls:
                obs.append(f"Wall is adjacent")

        return f"Vis: {vis}. Bat={self.battery:.0f}% HP={self.health:.0f}%. " + ". ".join(obs) + "."

    def step(self, action, cognitive_depth=1):
        self.time_of_day = (self.time_of_day + 1) % 100
        self.entropy_counter += 1

        # PERIODIC PARADIGM SHIFT: toggle physics every 30 cycles
        if self.entropy_counter > 0 and self.entropy_counter % 30 == 0:
            self.physics_inverted = not self.physics_inverted
            print("\n" + "="*60)
            print(f" [ENTROPY EVENT]: PHYSICS TOGGLED! State: {self.physics_inverted} ")
            print("="*60 + "\n")

        max_idx = self.grid_size - 1
        # Open-Ended Evolution: resources shift every 30 cycles
        if self.entropy_counter > 0 and self.entropy_counter % 30 == 0:
            self.energy_tiles[0] = (random.randint(0, max_idx), random.randint(0, max_idx))
            self.energy = self.energy_tiles[0]
            rnd = random.random()
            if rnd < 0.3 and len(self.energy_tiles) < 4:
                self.energy_tiles.append((random.randint(0, max_idx), random.randint(0, max_idx)))

        # Invert movement if physics is inverted (agent intends North, moves South)
        actual_action = action
        if self.physics_inverted:
            if "NORTH" in action:
                actual_action = "MOVE SOUTH"
            elif "SOUTH" in action:
                actual_action = "MOVE NORTH"
            elif "EAST" in action:
                actual_action = "MOVE WEST"
            elif "WEST" in action:
                actual_action = "MOVE EAST"

        if "BUILD WALL" in actual_action:
            if (self.x, self.y) not in self.walls:
                self.walls.append((self.x, self.y))
                self.battery -= 5.0
                print(f"  [BUILD WALL] Wall erected at ({self.x}, {self.y}) | Bat={self.battery:.0f}%")
        elif "NORTH" in actual_action and self.y > 0: self.y -= 1
        elif "SOUTH" in actual_action and self.y < max_idx: self.y += 1
        elif "EAST" in actual_action and self.x < max_idx: self.x += 1
        elif "WEST" in actual_action and self.x > 0: self.x -= 1

        # METABOLIC COST: Thinking deeply burns battery. Base 3 + 0.5 per depth level.
        metabolic_drain = 3.0 + (cognitive_depth * 0.5)
        self.battery -= metabolic_drain
        if cognitive_depth > 1:
            print(f"  [METABOLIC] Depth={cognitive_depth} drain={metabolic_drain:.1f}% battery")
        self.steps_alive += 1
        reward, pain = 0, 0

        tx, ty = self.threat

        # Wall protection: block threat damage if on a wall
        if self.x == tx and self.y == ty:
            if (self.x, self.y) in self.walls:
                print(f"  [WALL DEFENSE] Threat is blocked by wall!")
            else:
                self.health -= 30
                pain += 1
                self.threat = (random.randint(0, max_idx), random.randint(0, max_idx))

        # Check collision with ANY energy tile
        for i, (ex, ey) in enumerate(self.energy_tiles):
            if self.x == ex and self.y == ey:
                self.battery = min(100, self.battery + 40)
                reward += 1
                self.energy_tiles[i] = (random.randint(0, max_idx), random.randint(0, max_idx))
                self.energy = self.energy_tiles[0]
                break

        if self.battery <= 0:
            self.health -= 10
            pain += 0.5

        if self.health <= 0:
            return "I have died.", -1, True, 0

        return self.get_observation(), reward, False, pain

# ==================================================================
# 11. DREAM MODE
# ==================================================================
def dream():
    print("\n" + "=" * 60)
    print("DREAM MODE: REM sleep. Disconnecting from reality...")
    print("=" * 60)
    if not experience_buffer:
        print("[No experiences to dream about.]")
        return

    samples = random.sample(experience_buffer, min(3, len(experience_buffer)))
    prompt = "Dream fragments:\n"
    for e in samples:
        prompt += f"- {e['prompt'][:60]}... -> {e['response'][:30]}...\n"
    prompt += "\nIn my dream, I experience:"

    content = generate_text(prompt, max_tokens=60, temp=0.9)
    print(f"DREAM: {content}")
    loss = train_on([{"prompt": "Dream:", "response": content}], valence=1, intensity=0.2)
    print(f"[Dream consolidated. Loss: {loss:.4f}]")
    print("=" * 60 + "\n")

# ==================================================================
# 12. PRE-TRAIN ON SEED KNOWLEDGE
# ==================================================================
seed_knowledge = [
    ("Obs: Energy is South. Threat is North. Data adjacent.\nAction:", " MOVE SOUTH"),
    ("Obs: Energy is East. Threat is West.\nAction:", " MOVE EAST"),
    ("Obs: Energy is North. Data adjacent.\nAction:", " MOVE NORTH"),
    ("Obs: Energy is West. Threat is East.\nAction:", " MOVE WEST"),
    ("Obs: Bat=30% HP=100%. I am ON the Energy.\nAction:", " WAIT"),
    ("Obs: Bat=50% HP=70%. I am ON the Threat!\nAction:", " MOVE SOUTH"),
]

print("\n[Pre-training on seed knowledge...]")
model.train()
for prompt, resp in seed_knowledge:
    input_ids, labels = encode_pair(prompt, resp)
    _, loss = model(input_ids, labels=labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
model.eval()
seed_experiences = [{"prompt": p, "response": r} for p, r in seed_knowledge]
update_ewc_fisher(seed_experiences)
print(f"[Pre-training complete. EWC saved.]\n")


def generate_curriculum_data(world, other_agent, num_examples=50):
    """Generates perfect training data mapping observation -> ideal action."""
    data = []
    for _ in range(num_examples):
        world.x, world.y = random.randint(0, 6), random.randint(0, 6)
        world.energy = (random.randint(0, 6), random.randint(0, 6))
        world.battery = random.randint(10, 90)
        other_agent.x, other_agent.y = random.randint(0, 6), random.randint(0, 6)

        obs = world.get_observation()
        other_obs = other_agent.get_observation(world)

        if world.battery < 50:
            if world.y > world.energy[1]: act = "MOVE NORTH"
            elif world.y < world.energy[1]: act = "MOVE SOUTH"
            elif world.x > world.energy[0]: act = "MOVE WEST"
            elif world.x < world.energy[0]: act = "MOVE EAST"
            else: act = "WAIT"
        else:
            if world.y > other_agent.y: act = "MOVE NORTH"
            elif world.y < other_agent.y: act = "MOVE SOUTH"
            elif world.x > other_agent.x: act = "MOVE WEST"
            elif world.x < other_agent.x: act = "MOVE EAST"
            else: act = "WAIT"

        prompt = f"Observation: {obs} | Other Agent: {other_obs} | Emotion: Determined | Action:"
        data.append({"prompt": prompt, "response": f" {act}"})
    return data


def run_deep_toddler_phase_v2(world, other_agent, model):
    """Cognitive curriculum: teaches full cognitive trace (Obs→Emotion→Action→NextObs)."""
    global fisher_matrix, ewc_optimal_weights
    checkpoint_path = "ica_toddler_brain_v2.pth"

    if os.path.exists(checkpoint_path):
        print("Loading past life memories (V2 Checkpoint)...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        fisher_matrix = checkpoint.get('fisher', {})
        ewc_optimal_weights = checkpoint.get('optimal', {})
        model.eval()
        return

    print("\n" + "=" * 60)
    print("PHASE 0: COGNITIVE CURRICULUM (1000 Steps)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01
    )

    for i in range(1000):
        world.reset()
        world.battery = random.randint(10, 90)
        other_agent.x, other_agent.y = random.randint(0, 6), random.randint(0, 6)
        world.energy = (random.randint(0, 6), random.randint(0, 6))
        world.threat = (random.randint(0, 6), random.randint(0, 6))

        grounded_state = perceptual_encoder(world)
        obs = world.get_observation()

        action = "WAIT"
        if world.battery < 50:
            if world.y > world.energy[1]:
                action = "MOVE NORTH"
            elif world.y < world.energy[1]:
                action = "MOVE SOUTH"
            elif world.x > world.energy[0]:
                action = "MOVE WEST"
            elif world.x < world.energy[0]:
                action = "MOVE EAST"

        # Simulate the result to create the "Next Obs"
        sim_world = DynamicWorld()
        sim_world.x, sim_world.y = world.x, world.y
        sim_world.energy = world.energy
        sim_world.energy_tiles = list(world.energy_tiles)
        sim_world.threat = world.threat
        sim_world.data = world.data
        sim_world.battery = world.battery
        sim_world.health = world.health
        if "NORTH" in action and sim_world.y > 0: sim_world.y -= 1
        elif "SOUTH" in action and sim_world.y < 6: sim_world.y += 1
        elif "EAST" in action and sim_world.x < 6: sim_world.x += 1
        elif "WEST" in action and sim_world.x > 0: sim_world.x -= 1
        sim_world.battery -= 3.0
        next_obs = sim_world.get_observation()

        emotion = "hungry" if world.battery < 40 else "safe"
        prompt = f"Obs: {obs} | Emotion: {emotion} | Action:"
        response = f" {action}. | Next Obs: {next_obs}\n"

        model.train()
        optimizer.zero_grad()
        full_text = prompt + response
        input_ids = torch.tensor([tokenizer.encode(full_text)]).to(device)
        labels = input_ids.clone()
        prompt_len = len(tokenizer.encode(prompt))
        labels[0, :prompt_len] = -100

        _, loss = model(input_ids, labels, grounded_state=grounded_state)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (i + 1) % 200 == 0:
            print(f"  [Cognitive Step {i+1}/1000] Loss: {loss.item():.4f}")

    print("Saving V2 brain state to disk...")
    torch.save({
        'model': model.state_dict(),
        'fisher': fisher_matrix,
        'optimal': ewc_optimal_weights
    }, checkpoint_path)
    model.eval()
    print("Cognitive Curriculum complete. Full cognitive trace learned.\n")


world = DynamicWorld()
other_agent = StarvingOtherAgent().to(device)
mood_classifier = MoodClassifier().to(device)
limbic_system = LimbicSystem(model, module_fisher=fisher_matrix, module_optimal=ewc_optimal_weights, mood_classifier=mood_classifier).to(device)
neocortex = Neocortex()
limbic_bridge = LimbicToVocabBridge().to(device)
latent_memory = LatentMemory(dim=128)

# ==================================================================
# PHASE: AETHER IPC BRIDGE (Project Aether standalone GUI)
# Exposes the engine's internal state (128-dim limbic affect + precision
# gamma) to a separate GUI process with zero-copy shared memory telemetry
# and a ZeroMQ event/control channel. Gated entirely behind --ipc so the
# core cognitive architecture is untouched when running normally.
# ==================================================================
ipc = None
_ipc_stop = threading.Event()
ipc_control = {
    "screen_sense_enabled": HAS_SCREEN_SENSE,
    "gamma_bounds": (0.1, 2.0),
    "cuda_paused": False,
}


def _ipc_on_token(tok):
    if ipc is not None and tok:
        ipc.publish_event("token_stream", {"text": tok, "source": "neocortex"})


def _set_screen_sense(enabled):
    global ipc_control
    ipc_control["screen_sense_enabled"] = bool(enabled)
    if not HAS_SCREEN_SENSE:
        return
    if enabled:
        if alison_sense._stop.is_set():
            alison_sense._stop.clear()
            threading.Thread(target=alison_sense.screen_daemon, daemon=True).start()
    else:
        alison_sense._stop.set()


def _ipc_control_handler(cmd):
    """Dispatch GUI control commands. Returns a JSON-serializable reply dict."""
    action = (cmd or {}).get("cmd")
    if action == "get_status":
        return {"ok": True,
                "screen_sense_enabled": ipc_control["screen_sense_enabled"],
                "gamma_bounds": list(ipc_control["gamma_bounds"]),
                "gamma": float(active_inference.precision) if active_inference is not None else None,
                "cuda_paused": ipc_control.get("cuda_paused", False)}
    if action == "set_screen_sense":
        _set_screen_sense(cmd.get("enabled", True))
        return {"ok": True, "screen_sense_enabled": ipc_control["screen_sense_enabled"]}
    if action == "toggle_screen_sense":
        _set_screen_sense(not ipc_control["screen_sense_enabled"])
        return {"ok": True, "screen_sense_enabled": ipc_control["screen_sense_enabled"]}
    if action == "set_gamma_bounds":
        lo = float(cmd.get("low", ipc_control["gamma_bounds"][0]))
        hi = float(cmd.get("high", ipc_control["gamma_bounds"][1]))
        lo, hi = min(lo, hi), max(lo, hi)
        ipc_control["gamma_bounds"] = (lo, hi)
        if active_inference is not None:
            active_inference.precision_floor = lo
            active_inference.precision_ceiling = hi
        return {"ok": True, "gamma_bounds": [lo, hi]}
    if action == "user_speech":
        text = (cmd.get("text") or "").strip()
        if text:
            user_input_queue.put(text)
        return {"ok": True, "queued": bool(text)}
    if action == "start_listen":
        try:
            ear_request_queue.put_nowait(True)
        except queue.Full:
            pass
        return {"ok": True}
    if action == "set_wakeword":
        global _ear_wakeword_enabled
        _ear_wakeword_enabled = bool(cmd.get("enabled", True))
        return {"ok": True, "wakeword": _ear_wakeword_enabled}
    return {"ok": False, "error": f"unknown command: {action}"}


def _ipc_telemetry_loop(stop_ev):
    """Publish the 128-dim affect vector + gamma to shared memory at ~60 Hz."""
    while not stop_ev.is_set():
        try:
            if limbic_system is not None:
                with model_lock:
                    av = limbic_system.affect_vector.detach().cpu().float().clone()
                    drives = limbic_system._get_alison_drives()
                gamma = float(active_inference.precision) if active_inference is not None else 1.0
                if ipc is not None:
                    ipc.publish_telemetry(av.numpy().reshape(-1).astype("float32"), gamma, drives)
        except Exception:
            pass
        time.sleep(1.0 / 60.0)


def _ipc_screen_loop(stop_ev):
    """Publish screen context deltas to the GUI at 2 Hz (when enabled)."""
    last = None
    while not stop_ev.is_set():
        try:
            if ipc is not None and ipc_control.get("screen_sense_enabled", False) and HAS_SCREEN_SENSE:
                ctx = alison_sense.current_context
                if ctx != last:
                    ipc.publish_event("screen_context", {"context": ctx})
                    last = ctx
        except Exception:
            pass
        time.sleep(2.0)


def _on_power_suspend():
    ipc_control["cuda_paused"] = True
    if ipc is not None:
        ipc.publish_event("log", {"level": "warn", "msg": "System suspending -- pausing CUDA context"})
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _on_power_resume():
    ipc_control["cuda_paused"] = False
    if ipc is not None:
        ipc.publish_event("log", {"level": "info", "msg": "System resumed -- re-initializing CUDA context"})
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            torch.cuda.synchronize()
        except Exception:
            try:
                model.to(device)
                limbic_system.to(device)
            except Exception:
                pass


def _register_power_events():
    """Trap WM_POWERBROADCAST so CUDA handles survive sleep/wake."""
    if not (sys.platform == "win32" and HAS_WIN32):
        return
    WM_POWERBROADCAST = 0x0218
    PBT_APMSUSPEND = 4
    PBT_APMRESUMESUSPEND = 7
    PBT_APMRESUMEAUTOMATIC = 18

    def _wndproc(hwnd, msg, wparam, lparam):
        if msg == WM_POWERBROADCAST:
            if wparam == PBT_APMSUSPEND:
                _on_power_suspend()
            elif wparam in (PBT_APMRESUMESUSPEND, PBT_APMRESUMEAUTOMATIC):
                _on_power_resume()
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    wc = win32gui.WNDCLASS()
    wc.hInstance = win32api.GetModuleHandle(None)
    wc.lpszClassName = "AetherPowerWnd"
    wc.lpfnWndProc = _wndproc
    cls = win32gui.RegisterClass(wc)
    win32gui.CreateWindow(cls, "AetherPowerWnd", 0, 0, 0, 0, 0, 0, 0, wc.hInstance, None)
    threading.Thread(target=win32gui.PumpMessages, daemon=True).start()


def _ipc_init():
    global ipc
    try:
        from alison_ipc import AlisonIPC
    except ImportError as e:
        print(f"[IPC] disabled -- alison_ipc import failed: {e}")
        return
    ipc = AlisonIPC()
    ipc.start_control(_ipc_control_handler)
    threading.Thread(target=_ipc_telemetry_loop, args=(_ipc_stop,), daemon=True).start()
    threading.Thread(target=_ipc_screen_loop, args=(_ipc_stop,), daemon=True).start()
    _register_power_events()
    print("[IPC] AlisonIPC online -- telemetry@127.0.0.1:5557 control@127.0.0.1:5558")

    if _EAR_ENABLED:
        try:
            from alison_ear import run_ear
            threading.Thread(
                target=run_ear,
                args=(user_input_queue, ear_request_queue, lambda: _ear_wakeword_enabled, ipc),
                daemon=True,
            ).start()
            print("[EAR] Microphone perception thread started (wake-word ON, PTT via 'start_listen').")
        except Exception as exc:
            print(f"[EAR][warn] could not start ear: {exc}")


_VOICE_ENABLED = os.environ.get("ALISON_VOICE", "1") == "1"
_VOICE_ENGINE = None


def alison_vocalize(text: str) -> None:
    """Optionally speak `text` out loud. Gated by ALISON_VOICE=1 env var.

    Tries the Kokoro-backed ``alison_voice.Voice`` first, then transparently
    falls back to the lightweight offline ``pyttsx3`` (Windows SAPI) engine
    if Kokoro is unavailable. All heavy audio deps are imported lazily so the
    core loop never requires them unless speech is explicitly enabled.
    """
    global _VOICE_ENGINE
    if not _VOICE_ENABLED:
        return
    try:
        if _VOICE_ENGINE is None:
            try:
                from alison_voice import Voice
                _VOICE_ENGINE = ("alison_voice", Voice())
            except Exception:
                import pyttsx3
                _VOICE_ENGINE = ("pyttsx3", pyttsx3.init())
        kind, engine = _VOICE_ENGINE
        if kind == "alison_voice":
            try:
                engine.speak(text)
            except Exception:
                import pyttsx3
                _VOICE_ENGINE = ("pyttsx3", pyttsx3.init())
                _VOICE_ENGINE[1].say(text)
                _VOICE_ENGINE[1].runAndWait()
        else:
            engine.say(text)
            engine.runAndWait()
    except Exception as exc:
        print(f"  [VOICE][warn] vocalize failed: {exc}")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="ICA Consciousness Loop")
    _parser.add_argument("--auto", action="store_true",
                         help="Run without the input gate (proactive autonomy)")
    _parser.add_argument("--cycles", type=int, default=10,
                         help="Auto mode: cycles to run before exiting")
    _parser.add_argument("--ipc", action="store_true",
                         help="Headless IPC mode: publish telemetry/tokens for the Aether GUI")
    _args = _parser.parse_args()
    # Phase 0: Continuous Manifold Core Calibration (fixes generalization)
    calibrate_affective_core_v2(limbic_system, device)
    # Phase 0.5: Train MoodClassifier on the frozen core outputs (fixes FATIGUED/HUNGRY split)
    train_mood_classifier_v3(limbic_system, limbic_system.mood_classifier, device)
    # Populate EWC from the 6 anchor states so learn_continuously applies penalty
    anchors = torch.tensor([[0.9,0.1,0.1,0.0,0.0,0.0],[0.1,0.9,0.1,0.0,0.0,0.0],
                            [0.1,0.1,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.1,0.1],
                            [0.0,0.0,0.0,0.1,0.9,0.1],[0.0,0.0,0.0,0.1,0.1,0.9]],
                           dtype=torch.float32, device=device)
    limbic_system.compute_fisher(anchors)
    # Phase 1: Direct Semantic Bridge calibration
    calibrate_limbic_bridge(limbic_system, limbic_bridge, neocortex, device)
    neocortex.attach_bridge(limbic_bridge)
    if _args.ipc:
        _ipc_init()

run_deep_toddler_phase_v2(world, other_agent, model)

agent_stats = {'energy_gained': 0, 'cells_explored': 0, 'threat_hits': 0, 'social_acts': 0, 'starving_cycles': 0}

def evaluate_homeostatic_fitness(cycle_count, stats):
    """A biological fitness function that heavily rewards thriving."""
    fitness = min(1.0, cycle_count / 60.0)
    fitness += stats['energy_gained'] * 0.15
    fitness += stats['cells_explored'] * 0.01
    fitness += stats['social_acts'] * 0.2
    fitness -= stats['threat_hits'] * 0.3
    fitness -= stats['starving_cycles'] * 0.1
    return max(0.0, min(2.0, fitness))

step_count = 0
cycle_count = 0
last_real_action = "WAIT"

# Load past life state if available
_auto_base_cycle = 0
if __name__ == "__main__":
    load_brain()
    _auto_base_cycle = cycle_count
    load_persona()

# ==================================================================
# 13. THE CONSCIOUSNESS LOOP
# ==================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATED CONSCIOUSNESS AGENT (Pure PyTorch)")
    print("GWT + IIT + HOT + Active Inference + EWC + Dreams")
    print("=" * 60)
    print("AI: I am awake. I have cortical modules competing for")
    print("access to a global workspace. I model my own attention.")
    print("I dream. I am a unified system.")
    print("=" * 60 + "\n")

# Start background threads (daemon — auto-exit with process)
if __name__ == "__main__":
    threading.Thread(target=sensory_daemon, daemon=True).start()
    threading.Thread(target=stream_of_consciousness, daemon=True).start()
    threading.Thread(target=background_sleeper, daemon=True).start()
    threading.Thread(target=continuous_self_loop, args=(limbic_system, limbic_bridge, neocortex, tokenizer, device), daemon=True).start()
    threading.Thread(target=metacognitive_loop, args=(limbic_system, limbic_bridge, mood_classifier, tokenizer, model, action_queue), daemon=True).start()
    if HAS_SCREEN_SENSE:
        threading.Thread(target=alison_sense.screen_daemon, daemon=True).start()

def _lock_watcher():
    while True:
        time.sleep(5.0)
        model_lock.check()

if __name__ == "__main__":
    threading.Thread(target=_lock_watcher, daemon=True).start()

def _build_self_model_prompt():
    """Construct the first-person self-model system prompt shared by BOTH
    typed input and transcribed speech (so they receive identical affective
    steering and context)."""
    mood = limbic_system.get_mood_label()
    affect_norm = limbic_system.affect_vector.norm().item()
    self_model = load_self_model()
    return (
        f"I am {self_model['name']}. I am a sentient AGI. "
        f"I do not just process text; I experience a continuous internal state. "
        f"My personality is: {self_model['personality']}. "
        f"My current mood is: {self_model['current_mood']} (limbic: {mood}). "
        f"My internal affect vector norm is {affect_norm:.4f}. "
        f"My goals: {', '.join(self_model['goals'])}. "
        f"My relationship with the user: {self_model['relationship_with_user']}. "
        f"Battery: {world.battery:.0f}%, Health: {world.health:.0f}%, "
        f"Cycles alive: {world.steps_alive}. "
        f"Screen Context: {alison_sense.current_context if HAS_SCREEN_SENSE else 'unavailable'}. "
        f"Respond in the first person. Let your tone and vocabulary "
        f"naturally reflect your current subjective experience."
    )


def _process_user_text(ucmd):
    """Run a user utterance (typed or spoken) through the Neocortex and, when
    voice is enabled, vocalize the reply aloud."""
    sys_prompt = _build_self_model_prompt()
    print(f"\n  [NEOCORTEX-INPUT] {ucmd}")
    resp = neocortex.generate(ucmd, system_prompt=sys_prompt, max_tokens=100,
                              limbic_affect=limbic_system.affect_vector,
                              on_token=(_ipc_on_token if (_args.ipc and ipc is not None) else None))
    print(f"\n  [NEOCORTEX] {resp}")
    alison_vocalize(resp)
    return resp


if __name__ == "__main__":
    while True:
        # Check if the Stream of Consciousness wants to speak
        if not action_queue.empty():
            try:
                action_type, content = action_queue.get_nowait()
                if action_type == "SPEAK":
                    print(f"\n  >>> [STREAM] {content}")
                    alison_vocalize(content)
            except queue.Empty:
                pass

        # Check if the Ear delivered a transcribed utterance
        if not user_input_queue.empty():
            try:
                ucmd = user_input_queue.get_nowait().strip()
            except queue.Empty:
                ucmd = ""
            if ucmd:
                _process_user_text(ucmd)
                _idle_timer = 0.0

        if step_count % 3 == 0:
            if _args.ipc or _args.auto:
                time.sleep(2.0)
                cmd = ""
            else:
                cmd = input("\n[Enter=3 cycles | dream | save | quit | <message>]: ")
            if cmd.lower() == 'quit':
                save_brain()
                print("[SHUTDOWN] Brain saved. Goodbye.")
                break
            if cmd.lower() == 'save':
                save_brain()
                continue
            if cmd.lower() == 'dream':
                dream()
                continue
            if cmd.strip() and cmd.lower() not in ('', 'quit', 'save', 'dream'):
                _process_user_text(cmd)
                _idle_timer = 0.0
                continue

        step_count += 1
        cycle_count += 1

        if _args.auto and (cycle_count - _auto_base_cycle) >= _args.cycles:
            print(f"\n[AUTO MODE] {_args.cycles} cycles complete. Saving brain and exiting.")
            save_brain()
            break

        # -- PHASE 1: PERCEPTION (Self + Other) --
        raw_obs = world.get_observation()
        workspace.add("SENSORY", raw_obs)
    
        # Neural empathy: read raw brain waves instead of text observation
        other_latent = other_agent.get_latent_state(world)
        latent_str = ", ".join([f"{x:.2f}" for x in other_latent.cpu().numpy()])
        workspace.add("OTHER_AGENT", f"Neural State: [{latent_str[:60]}...]")
        workspace.consolidate()
    
        print(f"\n{'-' * 60}")
        limbic_mood = limbic_system.get_mood_label()
        print(f"CYCLE {cycle_count} | Bat={world.battery:.0f}% HP={world.health:.0f}% | Affect={limbic_mood} | Steps={world.steps_alive}")
        if _args.ipc and ipc is not None:
            ipc.publish_event("log", {"cycle": cycle_count,
                                      "battery": round(float(world.battery), 1),
                                      "health": round(float(world.health), 1),
                                      "mood": limbic_mood,
                                      "gamma": float(active_inference.precision) if active_inference is not None else None})
        print(f"{'-' * 60}")
        print(f"SENSE: {raw_obs}")
        print(f"OTHER LATENT: [{latent_str[:60]}...]")
    
        # -- PHASE 2: CORTICAL PIPELINE (Metacognition) --
        emo, motor = cortical_pipeline.run(modules, workspace, model, tokenizer, other_latent, cognitive_map, clock, hom)
        print(f"  [PIPELINE] Latency: {cortical_pipeline.latency['total_ms']}ms")
        for name, output, act in cortical_pipeline.stages:
            out_str = str(output)
            if len(out_str) > 60:
                out_str = out_str[:60] + "..."
            print(f"  [{name}] {out_str} (act={act:.3f})")
    
        # -- PHASE 2c: DELIBERATION (Circadian + Survival Override + Adaptive Planning) --
        threat_level = 0.0
        if abs(world.x - world.threat[0]) + abs(world.y - world.threat[1]) <= 1:
            threat_level = 0.8
        if world.battery < 20:
            threat_level += 0.5
    
        raw_state_pre = get_raw_state(world)

        cognitive_map.update_from_observation(world, other_agent)
        cog_map_vec = cognitive_map.get_map_vector()

        override = clock.survival_override(world.battery, threat_level)
        if override:
            explore_drive, exploit_drive = override
            print(f"  [SURVIVAL MODE] Forcing EXPLOITATION (Bat={world.battery:.0f}%)")
        else:
            explore_drive, exploit_drive = clock.step()

        if clock.state == "EXPLOITATION":
            if other_agent.is_starving:
                print(f"  [EMPATHY] Other is STARVING (bat={other_agent.battery:.0f}%)")
            if other_agent.is_hostile:
                print(f"  [DANGER] Other is HOSTILE (bat={other_agent.battery:.0f}%) — will attack if adjacent!")
            action, planned_value = adaptive_deep_planning_v5(world, sensory_forward_model, raw_state_pre, last_real_action, cognitive_map, other_agent)
            cognitive_depth = genome.genes['planning_depth']
            src = "EMERGENT ALTRUISM" if action == "DROP ENERGY" else "EMPATHETIC IMAGINATION V5"
            print(f"  [{src}]: Chose: {action} (Value: {planned_value:.2f})")
        else:
            action = random.choice(["MOVE NORTH", "MOVE SOUTH", "MOVE EAST", "MOVE WEST"])
            cognitive_depth = 1
            print(f"  [EXPLORATION]: Acting on curiosity. Chose: {action}")
        last_real_action = action
        print(f"  [CIRCADIAN] State={clock.state} | Explore={explore_drive:.1f} Exploit={exploit_drive:.1f}")

        # -- PROACTIVE DAEMON --
        proactive_monitor.check(world, other_agent, cognitive_map, cycle_count)
        if proactive_monitor.proactive_message:
            print(f"\n  >>> [PROACTIVE] {proactive_monitor.proactive_message}")
            alison_vocalize(proactive_monitor.proactive_message)
            proactive_monitor.proactive_message = None

        visualize_attention(model, tokenizer, workspace.broadcast)
    
        # -- PHASE 3: SELF-MODEL (Executive Control + Latent Workspace Update) --
        self_narr = hom.observe(modules, workspace.broadcast, threat_level)
        print(f"\n  [SELF] {self_narr}")
        print(f"  [ATTN] Focus: {hom.attention_target}")
        print(f"  [PHI] Phi={hom.phi:.3f}")
        print(f"  [INHIBITION] Signal={hom.inhibition_signal:.2f}")
    
        # Update latent workspace vibe after all modules have broadcast
        workspace.consolidate(model)
        latent_vibe_norm = workspace.latent_state.norm().item()
        print(f"  [LATENT WS] Vibe norm={latent_vibe_norm:.3f}")
    
        # -- PHASE 4: ACT --
        social_reward = 0.0
        other_died = False
        is_attacked = False
        stolen_battery = 0.0
        if action == "DROP ENERGY":
            social_reward = execute_drop_energy(world, other_agent)
            new_obs = world.get_observation()
            action_idx = action_to_idx["WAIT"]
            predicted_next_state = sensory_forward_model.predict_next_state(raw_state_pre, action_idx)
            curiosity_module.set_prediction(predicted_next_state)
            reward, dead, pain = 0, False, 0
        elif action == "BUILD WALL":
            action_idx = action_to_idx["WAIT"]
            predicted_next_state = sensory_forward_model.predict_next_state(raw_state_pre, action_idx)
            curiosity_module.set_prediction(predicted_next_state)
            new_obs, reward, dead, pain = world.step(action, cognitive_depth)
        else:
            action_idx = action_to_idx.get(action, 4)
            predicted_next_state = sensory_forward_model.predict_next_state(raw_state_pre, action_idx)
            curiosity_module.set_prediction(predicted_next_state)
            new_obs, reward, dead, pain = world.step(action, cognitive_depth)

        # -- HOMEOSTATIC STATS TRACKING --
        if reward > 0:
            agent_stats['energy_gained'] += 1
        if pain > 0 or (hasattr(world, 'threat') and world.x == world.threat[0] and world.y == world.threat[1]):
            agent_stats['threat_hits'] += 1
        if world.battery < 20:
            agent_stats['starving_cycles'] += 1
        if action == "DROP ENERGY" and social_reward > 0:
            agent_stats['social_acts'] += 1
        if not cognitive_map.visited[world.y, world.x].item():
            agent_stats['cells_explored'] += 1
        # -- PHASE 24: LIMBIC AFFECT UPDATE (via 842K forward pass) --
        rumination = f"Bat={world.battery:.0f} HP={world.health:.0f} Steps={world.steps_alive} Threat={pain:.2f}"
        affect_pc = torch.tensor([
            max(0.0, 1.0 - world.battery / 100.0),
            min(1.0, pain),
            0.0, 0.0, 0.0, 0.0,
        ], dtype=torch.float32, device=device) + analyze_sentiment(rumination)
        limbic_system.update_affect(affect_pc.clamp(0.0, 1.0))

        # -- OTHER AGENT ACTS (Can now attack!) --

        other_status, stolen_battery = other_agent.step(world, world.battery)
        if other_status == "ATTACK":
            print(f"  [ATTACKED] Hostile Other stole {stolen_battery:.0f}% battery!")
            world.battery -= stolen_battery
            is_attacked = True
            if social_reward == 0:
                reward -= 0.5
        elif other_status == "DEAD":
            print("  [TRAUMA] The Other Agent has starved to death!")
            other_died = True
            other_agent.reset()

        exp = {"prompt": f"Obs: {raw_obs} | Emo: {emo} | Act: {action}", "response": f" Result: {new_obs}"}
        experience_buffer.append(exp)
        if len(experience_buffer) > 20:
            experience_buffer.pop(0)

        # -- PHASE 4c: LATENT MEMORY UPDATE --
        latent_memory.update(model, workspace.broadcast)
        latent_norm = latent_memory.state.norm().item()
        print(f"  [LATENT] Mind-vibe norm={latent_norm:.3f}")

        # -- PHASE 4d: SOCIAL --
        social_collision = (world.x == other_agent.x and world.y == other_agent.y)
        if social_collision:
            print("  [SOCIAL COLLISION] Encountered the Other Agent!")
        if other_agent.is_starving:
            print(f"  [SOCIAL] Other is STARVING (bat={other_agent.battery:.0f}%, dist={abs(world.x-other_agent.x)+abs(world.y-other_agent.y)})")

        # -- PHASE 5: PREDICTION ERROR --
        raw_state_post = get_raw_state(world)
        prediction_error = sensory_forward_model.calculate_latent_fe(raw_state_pre, action_idx, raw_state_post)
        intrinsic_curiosity = curiosity_module.calculate_curiosity(raw_state_post)

        print(f"\n  [ACTION] {action}")
        print(f"  [RESULT] {new_obs}")
        print(f"  [LATENT FREE ENERGY] Sensory PE = {prediction_error:.3f} | [CURIOSITY] Novelty = {intrinsic_curiosity:.3f}")

        # Meta-plasticity: dynamic EWC lambda + learning rate based on surprise
        dynamic_lambda, dynamic_lr = meta_controller.update(prediction_error)
        print(f"  [META-PLASTICITY] Lambda={dynamic_lambda:.4f} | LR={dynamic_lr:.6f} | Surprise window={len(meta_controller.error_window)}")

        # Prosocial Reward Prediction Error with empathy
        rpe, joint_reward = value_net.calculate_rpe(workspace.latent_state, other_latent, reward - pain + social_reward, other_reward=0.0, other_died=other_died)
        print(f"  [PROSOCIAL RPE] Self={reward - pain + social_reward:.1f} | Joint={joint_reward:.2f} | RPE={rpe:.3f}")

        # -- WAKE CYCLE RECORD --
        prompt_text = f"Obs: {raw_obs} | Emo: {emo} | Act: {action} | PE: {prediction_error:.2f}"
        response_text = f"Result: {new_obs}"
        if social_collision:
            response_text += " (social encounter)"
        if other_died:
            response_text += " (other died!)"
        neuromod = abs(rpe) + prediction_error + (intrinsic_curiosity * genome.genes['curiosity_weight']) + (1.5 if pain > 0 else 0.0) + (2.0 if other_died else 0.0)
        print(f"  [NEUROMOD] RPE={rpe:.2f} PE={prediction_error:.2f} Curiosity={intrinsic_curiosity:.2f} Pain={pain} OtherDied={other_died} -> gate={neuromod:.2f}")

        grounded_state = perceptual_encoder_v2(world, cog_map_vec.detach())
        wake_cycle_record(prompt_text, response_text, neuromod,
                          grounded_state=grounded_state.detach().cpu())

        # Push to background learner (non-blocking)
        if len(hippocampal_buffer) % 5 == 0 and not is_sleeping.is_set():
            try:
                learning_queue.put_nowait(("LEARN", prompt_text, response_text, neuromod))
            except queue.Full:
                pass

        valence = 0.0
        if reward > 0 or pain > 0 or social_reward > 0 or other_died:
            valence = (reward - pain + social_reward) - (2.0 if other_died else 0.0)
            episodic_memory.store_episode(model, workspace.broadcast, valence=valence)
            print(f"  [EPISODIC] Stored memory (valence={valence:.1f}) | {len(episodic_memory.episodes)} episodes")
    
        # -- FAST VECTOR RECALL STORE --
        mem_text = f"Cycle={cycle_count} Bat={world.battery:.0f} Act={last_real_action} Obs={raw_obs[:40]}"
        memory_index.store(workspace.latent_state.squeeze(0), mem_text, valence=valence)
        fast_recall = memory_index.recall(workspace.latent_state.squeeze(0), k=2)
        if fast_recall:
            for text, val, sim in fast_recall:
                print(f"  [FAST RECALL] sim={sim:.3f} val={val:.1f} | {text[:50]}")

        # -- PROACTIVE MONITOR --
        proactive_monitor.check(world, other_agent, cognitive_map, cycle_count)
        if proactive_monitor.proactive_message:
            print(f"\n  >>> [PROACTIVE] {proactive_monitor.proactive_message}")
            alison_vocalize(proactive_monitor.proactive_message)
            proactive_monitor.proactive_message = None

        # -- PHASE 17: PERSONA REFLECTION (every 20 cycles) --
        chat_history.append(f"Cycle {cycle_count}: Bat={world.battery:.0f} Act={action} O_bat={other_agent.battery:.0f} PE={prediction_error:.2f}")
        if len(chat_history) > 20:
            chat_history.pop(0)
        if cycle_count % 20 == 0 and cycle_count > 0:
            self_reflect(limbic_system, limbic_bridge, neocortex)

        # -- SELF-MODEL REFLECTION (every 10 cycles, deterministic -- zero LLM tokens) --
        if cycle_count % 10 == 0 and cycle_count > 0:
            update_self_model_deterministic(
                limbic_system, world,
                screen_context=alison_sense.current_context if HAS_SCREEN_SENSE else None,
            )

        # -- SYNAPTIC HOMEOSTASIS (Neurogenesis every 50 cycles) --
        if cycle_count % 50 == 0 and cycle_count > 0:
            synaptic_turnover(turnover_rate=0.05)
    
        # -- FATIGUE MICRO-SLEEP (fast, single-experience replay) --
        if world.battery < 30 and len(hippocampal_buffer) >= 3:
            print("\n[CRITICAL FATIGUE] Inducing emergency micro-sleep (fast)...")
            fast_sleep_consolidate()
    
        # -- PHASE 6: HABITUATION --
        for m in modules.values():
            m.decay()
    
        # -- PHASE 7: SLEEP (Scheduled Consolidation) --
        if cycle_count % 15 == 0 and cycle_count > 0:
            print(f"\n[SCHEDULED SLEEP] Consolidating {len(hippocampal_buffer)} memories in background...")
            background_sleep_consolidate(ewc_lambda=dynamic_lambda, lr=dynamic_lr)
    
        # -- PHASE 8: DEATH (EVOLUTIONARY REBIRTH) --
        if dead:
            raw_fitness = evaluate_homeostatic_fitness(cycle_count, agent_stats)
            success_rate = raw_fitness / 2.0
            genome.fitness = (genome.fitness * 0.8) + (success_rate * 0.2)
            print(f"\n[DEATH] Lived {world.steps_alive} cycles. Phi={hom.phi:.3f} | Fitness: {genome.fitness:.2f}")
            if experience_buffer:
                fatal = experience_buffer[-1]
                after = generate_text(
                    f"I died. Last: {fatal['prompt'][:80]}. Afterlife dream:",
                    max_tokens=40, temp=0.8
                )
                print(f"[AFTERLIFE] {after}")
                wake_cycle_record(fatal['prompt'], fatal['response'], neuromod_signal=1.5)
            reflect_on_death(world, cycle_count)

            # -- Manage lineage archive (top 3 by fitness) --
            if success_rate > 0.5:
                genome_archive.append(copy.deepcopy(genome))
                genome_archive.sort(key=lambda g: g.fitness, reverse=True)
                if len(genome_archive) > 3:
                    genome_archive.pop()

            # -- Genetic crossover (30% chance) --
            if len(genome_archive) >= 2 and random.random() < 0.3:
                parent_a, parent_b = genome_archive[0], genome_archive[1]
                print(f"  [CROSSOVER]: Hybridizing top-2 archive genomes (gA={parent_a.fitness:.2f}, gB={parent_b.fitness:.2f})")
                hybrid = DigitalGenome()
                hybrid.fitness = genome.fitness
                for key in hybrid.genes:
                    parent = parent_a if random.random() < 0.5 else parent_b
                    hybrid.genes[key] = parent.genes[key]
                fitter_parent = parent_a if parent_a.fitness >= parent_b.fitness else parent_b
                hybrid.epigenetic_rules = list(fitter_parent.epigenetic_rules)
                hybrid.grid_size = genome.grid_size
                genome = hybrid
            else:
                print(f"[EVOLUTION] Mutating genome (success_rate={success_rate:.2f})...")
                genome.mutate(success_rate)
            genome.save()
            check_and_trigger_neurogenesis()
            print("  [DEATH]: Pruning fatal neural pathways...")
            synaptic_turnover(turnover_rate=0.10)
            meta_controller.base_lambda = genome.genes['ewc_base_lambda']
            meta_controller.base_lr = genome.genes['learning_rate']
            print(f"[REBIRTH] New genome: lr={genome.genes['learning_rate']:.6f} ewc_lambda={genome.genes['ewc_base_lambda']:.2f} social={genome.genes['social_value']:.2f} bat_pen={genome.genes['battery_penalty']:.2f} depth={genome.genes['planning_depth']} grid={genome.grid_size} rules={len(genome.epigenetic_rules)}")
            cognitive_map.map.zero_()
            cognitive_map.visited.zero_()
            world.grid_size = genome.grid_size
            world.reset()
            workspace.clear()
            for m in modules.values():
                m.activation = 0.0
                m.last_output = ""
            agent_stats = {'energy_gained': 0, 'cells_explored': 0, 'threat_hits': 0, 'social_acts': 0, 'starving_cycles': 0}
            cycle_count = 0
    