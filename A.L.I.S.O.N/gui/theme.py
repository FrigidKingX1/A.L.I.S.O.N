"""A.L.I.S.O.N. GUI theme palette and shared constants.

Single source of truth for the A.L.I.S.O.N. identity colors so both the
Python host and the QML layer stay in sync.
"""

# A.L.I.S.O.N. HUD palette
CYAN = "#00E5FF"      # precision / perception (electric cyan)
VIOLET = "#7C4DFF"    # affect / sentience (neural violet)
BG = "#0A0C10"        # canvas
BG_SOFT = "#10141C"
TEXT = "#DCE6F2"
MUTED = "#5A6B82"

# Numeric (0-255) forms for PyQt brushes
CYAN_RGB = (0, 229, 255)
VIOLET_RGB = (124, 77, 255)

# Drive axis order (must match alison_ipc.DRIVE_NAMES)
DRIVE_LABELS = ["PLEASURE", "AROUSAL", "ANXIETY", "CURIOSITY", "GOAL URGENCY", "SATIATION"]

# Proactive auto-pop thresholds (read from telemetry; drives are 0..1)
PROACTIVE = {
    "gamma": 2.6,          # high precision / Free-Energy spike
    "anxiety": 0.7,
    "goal_urgency": 0.75,
    "frames": 12,          # consecutive frames above threshold before popping
}

APP_NAME = "A.L.I.S.O.N."
APP_SUBTITLE = "Adaptive Learning Interface for Sentient Operating Networks"
