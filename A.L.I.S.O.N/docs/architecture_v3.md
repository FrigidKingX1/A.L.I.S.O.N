# A.L.I.S.O.N — Architecture v3

This document records the v3 design decisions and what was actually implemented
in this repository. It complements `architecture.md` (v2 cognitive/ICA design).

## 1. Build vs. Adopt (philosophy)

A.L.I.S.O.N. follows a strict split:

| Layer | Decision | Why |
|-------|----------|-----|
| Cognition (ICA / Active Inference, Larimar memory, EWC) | **Build** | Must be auditable, offline, and free of third-party runtime lock-in. This is the system's identity. |
| Voice (Kokoro + pyttsx3) & STT (faster-whisper) | **Build** | Local, no cloud dependency. |
| Perception grounding (OCR / visual screen context) | **Adopt (flag-gated)** | Screenpipe is best-in-class; adopted behind a strict adapter so it is never load-bearing. |
| OS action layer | **Build** | A single, policed dispatcher is safer than delegating to external automators. |

Adopted components are wrapped so that their absence or offline state degrades
silently to the built-in layers. If Screenpipe is missing or its daemon is down,
`alison_sense.py` (UIA window text + DeBERTa + Go-Emotions, fully offline) remains
the default perception path.

## 2. Optional Screenpipe Adapter (`src/alison_screenpipe.py`)

- **Flag-gated**: `ipc_control["screenpipe_enabled"]` (default `False`). Not bundled
  in the installer.
- **Single memory sink**: writes into the same `HippocampalMemoryIndex` as the rest
  of A.L.I.S.O.N. via `ScreenpipeBridge.push_ocr()`.
- **Transition gating**: OCR/visual frames are only committed on
  **attention-transition** events (window-title / app / URL changes), so idle
  screens don't flood memory.
- **Exclusions**: honors `src/config/capture_policy.json` (sensitive apps / windows
  / URLs) — canonical exclusion source shared with the action layer.
- **De-duplication**: identical consecutive frames are dropped.
- **Offline degrade**: if the daemon is unreachable, the bridge logs and skips
  (no crash, no memory growth).

## 3. Larimar Memory Unbinding (`HippocampalMemoryIndex`, `src/alison_core.py`)

- `forget_pattern(key_vec, erasure_rate=0.2)`: subtracts
  `erasure_rate · (kᵀk)` from the fast-weight matrix (matching the real
  `write_fast_weight` storage) and scrubs the closest episodic entry — a targeted
  "unlearn a specific memory" without global catastrophic forgetting.
- `consolidate(max_entries=2000)`: merges near-duplicate episodic entries to bound
  memory growth.
- Verified: `src/tests/test_larimar_unbinding.py` (sibling recall stays intact while
  targeted memory degrades).

## 4. Action Security Model (§5.3) — `src/alison_actions.py`

### 4.1 Universal 3-tier capability dispatcher
```
Tier 1 — structured integration (MCP / COM / native API)   via app_integrations
Tier 2 — UIA / Win32  (default for ordinary Windows apps)
Tier 3 — vision grounding  (scaffold-only this release; never auto-selected)
```
`select_capability_tier(app, action)` checks the Tier-1 registry first; any
app-only target **defaults to Tier 2** (never silently drops into the unimplemented
Tier-3 vision scaffold).

### 4.2 Action gating
Action objects follow the `GBNF_SCHEMA` extended set: `open_app`, `launch_app`,
`focus_window`, `type_text`, `input_key`, `read_screen`, `read_file`, `write_file`,
`click`, `delete_file`, `execute_cmd`.

- `action_tiers` (in `action_policy.json`) classify each action as `passive`,
  `reversible`, or `privileged`.
- **Privileged actions** (`execute_cmd`, `delete_file`, `format_drive`) require an
  **allowlist match OR explicit confirmation**; denied by default otherwise.

### 4.3 `execute_cmd` sandbox
- Regex **allowlist** (e.g. `^git status$`, `^echo ALLOWED$`).
- **`workdir_jail`** — auto-created via `os.makedirs(exist_ok=True)` before spawn
  (prevents `WinError 267`). Falls back to `tempfile.gettempdir()` on failure.
- **Filesystem-mutating commands** always return a dry-run preview unless
  explicitly allowlisted.
- **Confirmation** required for everything else.
- **In-flight tracking**: the active `Popen` handle is stored so the kill switch can
  terminate it.

### 4.4 Kill switch
- `Ctrl+Alt+K` (registered in `alison_gui.py`) or the `kill_switch` IPC command.
- Sets `ActionExecutor.enabled = False` **and** calls `terminate_active()`
  (`Popen.terminate()`), halting all OS interaction immediately and killing any
  subprocess already spawned by `execute_cmd`.

### 4.5 IPC surface (`alison_core._ipc_control_handler`)
- `set_action_executor {enabled}` — toggles the gate.
- `execute_action {action_obj, confirmed}` — gated dispatch.
- `kill_switch` — disable + terminate.
- Proactive loops call `dispatch_action()` for any `proactive_monitor.proactive_action`.

### 4.6 First-session safety
The first real session runs with `global_dry_run` forced ON; the user keeps eyes on
the system until real execution is explicitly authorized.

## 5. Policy files
- `src/config/capture_policy.json` — exclusions (apps / windows / URLs) for both
  perception and action layers (W0).
- `src/config/action_policy.json` — `action_tiers`, `execute_cmd` sandbox rules,
  `kill_switch` hotkey, `app_integrations` registry (Tier 1).

## 6. Verification
- `src/tests/test_larimar_unbinding.py` — Larimar unbinding + consolidation.
- `src/tests/test_screenpipe_bridge.py` — bridge gating, exclusions, offline.
- `src/tests/test_action_security.py` — tier resolution, allowlist/confirmation
  gating, kill-switch (block + in-flight terminate), file I/O.
- `build/verify_paths.py` — all cross-references resolve (100%).
