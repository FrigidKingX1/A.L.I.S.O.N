import re

lines = open("watchdog5.log", encoding="utf-16", errors="replace").read().splitlines()
print("LINES:", len(lines))
for i, l in enumerate(lines):
    s = l.strip()
    if re.search(
        r"Timeout|CYCLE \d|PIPELINE|PROACTIVE|AUTO MODE|METACOGNITION\] Cycle|"
        r"FAST RECALL|EPISODIC|ACTION\]|LOCK-WATCH|Traceback|NameError|RuntimeError|"
        r"File \"alison_core|in <module>",
        s,
    ):
        print(i, ":", s[:150])
