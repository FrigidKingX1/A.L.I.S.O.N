import re

lines = open("watchdog4.log", encoding="utf-16", errors="replace").read().splitlines()
for i, l in enumerate(lines):
    s = l.strip()
    if re.search(r"Timeout|Thread 0x|File \"alison_core|in <module>|LOCK-WATCH", s):
        print(i, ":", s[:160])