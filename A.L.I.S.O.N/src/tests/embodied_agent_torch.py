import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import re
import math

# ─── SIMPLE TOKENIZER ──────────────────────────────────────────────
class CharTokenizer:
    """Character-level tokenizer for the embodied world agent."""
    def __init__(self):
        self.bos = "<BOS>"
        self.eos = "<EOS>"
        self.pad = "<PAD>"
        self.unk = "<UNK>"
        # Build vocab from printable ASCII + special tokens
        chars = [chr(i) for i in range(32, 127)]
        specials = [self.bos, self.eos, self.pad, self.unk]
        self.stoi = {s: i for i, s in enumerate(specials)}
        for i, c in enumerate(chars):
            self.stoi[c] = len(self.stoi)
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.bos_id = self.stoi[self.bos]
        self.eos_id = self.stoi[self.eos]
        self.pad_id = self.stoi[self.pad]
        self.unk_id = self.stoi[self.unk]

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_id]
        for ch in text:
            ids.append(self.stoi.get(ch, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            ch = self.itos.get(i, self.unk)
            if ch in (self.bos, self.eos, self.pad):
                continue
            chars.append(ch)
        return "".join(chars)

tokenizer = CharTokenizer()

# ─── LoRA LAYER ────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 4.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora_a = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))
        self.alpha = alpha
        self.rank = rank
        self.linear.requires_grad_(False)

    def forward(self, x):
        base = self.linear(x)
        lora = (x @ self.lora_a @ self.lora_b) * (self.alpha / self.rank)
        return base + lora

    def merge(self):
        with torch.no_grad():
            weight_delta = (self.lora_a @ self.lora_b) * (self.alpha / self.rank)
            self.linear.weight.data.add_(weight_delta.T)
            self.lora_a.zero_()
            self.lora_b.zero_()

    def get_drift(self):
        return ((self.lora_a @ self.lora_b) * (self.alpha / self.rank)).norm().item()

# ─── TRANSFORMER BLOCK ─────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = LoRALinear(embed_dim, embed_dim)
        self.k_proj = LoRALinear(embed_dim, embed_dim)
        self.v_proj = LoRALinear(embed_dim, embed_dim)
        self.out_proj = LoRALinear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        att = att + mask
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        hidden = embed_dim * 4
        self.gate = LoRALinear(embed_dim, hidden)
        self.down = LoRALinear(hidden, embed_dim)

    def forward(self, x):
        return self.down(F.gelu(self.gate(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# ─── SMALL TRANSFORMER LM ──────────────────────────────────────────
class AgentLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 96, num_heads: int = 4,
                 num_layers: int = 4, max_seq_len: int = 256):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight  # weight tying
        self.max_seq_len = max_seq_len
        self.register_buffer('pos_embed', self._build_pos_encoding(max_seq_len, embed_dim))

    def _build_pos_encoding(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
        x = self.token_embed(input_ids) + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=40, temperature=0.6):
        device = next(self.parameters()).device
        curr = input_ids.clone().to(device)
        for _ in range(max_new_tokens):
            ctx = curr[:, -self.max_seq_len:]
            logits, _ = self.forward(ctx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            curr = torch.cat([curr, next_id], dim=1)
            if next_id.item() == tokenizer.eos_id:
                break
        return curr

# ─── BUILD MODEL ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = AgentLM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=96,
    num_heads=4,
    num_layers=4,
    max_seq_len=256,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# LoRA-only: freeze non-LoRA params
for name, p in model.named_parameters():
    if 'lora_' not in name:
        p.requires_grad_(False)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable (LoRA) parameters: {trainable:,}")

# ─── EWC STATE ─────────────────────────────────────────────────────
ewc_optimal = {}
ewc_fisher = {}

def save_optimal_weights():
    ewc_optimal.clear()
    ewc_fisher.clear()
    for name, p in model.named_parameters():
        if p.requires_grad:
            ewc_optimal[name] = p.detach().clone()
            ewc_fisher[name] = torch.ones_like(p) * 1.0

def update_fisher(loss):
    loss.backward(retain_graph=True)
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            fisher = p.grad.pow(2).detach()
            if name in ewc_fisher:
                ewc_fisher[name] = ewc_fisher[name] * 0.9 + fisher * 0.1
            else:
                ewc_fisher[name] = fisher

def ewc_penalty():
    if not ewc_optimal:
        return 0.0
    penalty = 0.0
    for name, p in model.named_parameters():
        if name in ewc_optimal and name in ewc_fisher:
            diff = p - ewc_optimal[name]
            pnl = (ewc_fisher[name] * diff.pow(2)).sum().item()
            penalty += pnl
    return penalty * 0.01

# ─── MEMORY & DOPAMINE ─────────────────────────────────────────────
action_history = []
memory_buffer = []  # persistent memory with importance

def encode(text):
    return torch.tensor([tokenizer.encode(text)])

def train_on_sequence(seq):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for step in seq:
        text = f"Obs: {step['user']}\nAction: {step['ai']}"
        input_ids = encode(text).to(device)
        labels = input_ids.clone()
        # mask prompt portion
        prompt_len = len(tokenizer.encode(f"Obs: {step['user']}\nAction: ")) - 1
        labels[:, :prompt_len] = -100
        _, loss = model(input_ids, labels=labels)
        loss.backward()
        total_loss += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    model.eval()
    return total_loss

def dopamine_learn(success_sequence):
    print("\n[DOPAMINE SPIKE: Reward achieved. Reinforcing neural pathways...]")
    loss = train_on_sequence(success_sequence[-3:])
    # add to persistent memory with boosted importance
    for step in success_sequence[-3:]:
        memory_buffer.append((step, 2.0))
    # update fisher
    model.train()
    input_ids = encode(f"Obs: {success_sequence[-1]['user']}\nAction: {success_sequence[-1]['ai']}").to(device)
    labels = input_ids.clone()
    prompt_len = len(tokenizer.encode(f"Obs: {success_sequence[-1]['user']}\nAction: ")) - 1
    labels[:, :prompt_len] = -100
    _, loss = model(input_ids, labels=labels)
    update_fisher(loss)
    optimizer.zero_grad()
    model.eval()
    print(f"[Reinforcement complete. Surprisal: {loss.item():.4f}]\n")
    save_optimal_weights()

# ─── PRIORITIZED REPLAY CONSOLIDATION ─────────────────────────────
def consolidate():
    if len(memory_buffer) < 3:
        return
    print("\n[Sleep: Consolidating memories...]")
    memory_buffer.sort(key=lambda x: x[1], reverse=True)
    top_memories = memory_buffer[:3]
    seq = [m[0] for m in top_memories]
    loss = train_on_sequence(seq)
    pnl = ewc_penalty()
    print(f"[EWC Penalty: {pnl:.4f}]")
    print(f"[Consolidation complete. Surprisal: {loss:.4f}]\n")

# ─── THE EMBODIED WORLD ────────────────────────────────────────────
class World:
    def __init__(self):
        self.grid_size = 5
        self.reset()

    def reset(self):
        self.ai_x, self.ai_y = random.randint(0, 4), random.randint(0, 4)
        self.energy_x, self.energy_y = random.randint(0, 4), random.randint(0, 4)
        self.data_x, self.data_y = random.randint(0, 4), random.randint(0, 4)
        self.battery = 50.0
        self.curiosity = 90.0

    def get_observation(self):
        obs = []
        if self.ai_y < self.energy_y: obs.append("Energy is to the South")
        elif self.ai_y > self.energy_y: obs.append("Energy is to the North")
        if self.ai_x < self.energy_x: obs.append("Energy is to the East")
        elif self.ai_x > self.energy_x: obs.append("Energy is to the West")

        if self.ai_y < self.data_y: obs.append("Data is to the South")
        elif self.ai_y > self.data_y: obs.append("Data is to the North")
        if self.ai_x < self.data_x: obs.append("Data is to the East")
        elif self.ai_x > self.data_x: obs.append("Data is to the West")

        if not obs: obs.append("You are standing on something.")
        return f"Battery: {self.battery:.1f}%, Curiosity: {self.curiosity:.1f}%. " + ". ".join(obs)

    def step(self, action):
        reward = 0
        if "NORTH" in action and self.ai_y > 0: self.ai_y -= 1
        elif "SOUTH" in action and self.ai_y < 4: self.ai_y += 1
        elif "EAST" in action and self.ai_x < 4: self.ai_x += 1
        elif "WEST" in action and self.ai_x > 0: self.ai_x -= 1

        self.battery -= 5.0
        self.curiosity -= 2.0

        if self.ai_x == self.energy_x and self.ai_y == self.energy_y:
            self.battery = min(100, self.battery + 40)
            reward += 1
            self.energy_x, self.energy_y = random.randint(0, 4), random.randint(0, 4)

        if self.ai_x == self.data_x and self.ai_y == self.data_y:
            self.curiosity = min(100, self.curiosity + 50)
            reward += 1
            self.data_x, self.data_y = random.randint(0, 4), random.randint(0, 4)

        if self.battery <= 0:
            return "You have run out of energy and shut down.", -1, True

        return self.get_observation(), reward, False

# ─── PRE-TRAIN ON SEED STRATEGIES ──────────────────────────────────
seed_strategies = [
    ("Energy is to the South. Data is to the East", "MOVE SOUTH"),
    ("Energy is to the North. Data is to the West", "MOVE NORTH"),
    ("Energy is to the East. Data is to the South", "MOVE EAST"),
    ("Energy is to the West. Data is to the North", "MOVE WEST"),
    ("You are standing on something.", "WAIT"),
]
print("\n[Pre-training on seed strategies...]")
model.train()
for obs, act in seed_strategies:
    text = f"Obs: {obs}\nAction: {act}"
    input_ids = encode(text).to(device)
    labels = input_ids.clone()
    prompt_len = len(tokenizer.encode(f"Obs: {obs}\nAction: ")) - 1
    labels[:, :prompt_len] = -100
    _, loss = model(input_ids, labels=labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
model.eval()
save_optimal_weights()
print("[Pre-training complete. EWC optimal weights saved.]\n")

# ─── AUTONOMOUS AGENT LOOP ────────────────────────────────────────
world = World()
print("AI: I am awake. I perceive a 2D space. I must survive and learn.")
model.eval()

step_count = 0

while True:
    if step_count % 10 == 0:
        cmd = input("\nPress Enter to let the AI live 10 steps, or type 'quit': ")
        if cmd.lower() == 'quit': break

    step_count += 1
    obs = world.get_observation()

    # 1. THINK (generate internal thought)
    thought_prompt = f"Obs: {obs}\nThought: I need to survive. My strategy is to"
    prompt_ids = encode(thought_prompt).to(device)
    thought_ids = model.generate(prompt_ids, max_new_tokens=20, temperature=0.6)
    thought = tokenizer.decode(thought_ids[0].tolist()).split("Thought:")[-1].strip().split("\n")[0]

    # 2. ACT (generate movement action)
    action_prompt = f"Obs: {obs}\nThought: {thought}\nAction: [MOVE"
    prompt_ids = encode(action_prompt).to(device)
    action_ids = model.generate(prompt_ids, max_new_tokens=5, temperature=0.2)
    action_raw = tokenizer.decode(action_ids[0].tolist()).split("Action:")[-1].strip()

    match = re.search(r'(NORTH|SOUTH|EAST|WEST)', action_raw.upper())
    action = f"[MOVE {match.group(1)}]" if match else "[WAIT]"

    print(f"\n--- Step {step_count} ---")
    print(f"World: {obs}")
    print(f"Thought: {thought}")
    print(f"Action: {action}")

    # 3. EXPERIENCE
    new_obs, reward, dead = world.step(action)
    action_history.append({"user": obs, "ai": action.replace("[", "").replace("]", "")})

    # 4. DOPAMINE / REWARD LEARNING
    if reward > 0:
        dopamine_learn(action_history)
        action_history.clear()
        print(f"[SYSTEM: Reward achieved! Battery: {world.battery:.1f}, Curiosity: {world.curiosity:.1f}]")

    # 5. PERIODIC CONSOLIDATION
    if step_count % 20 == 0:
        consolidate()

    if dead:
        print("[SYSTEM: AI died. Rebooting a new instance...]")
        world.reset()
        action_history.clear()
        consolidate()
