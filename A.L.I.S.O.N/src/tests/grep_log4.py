import re

lines = open("watchdog2.log", encoding="utf-16", errors="replace").read().splitlines()
print("LINES:", len(lines))
for i, l in enumerate(lines):
    s = l.strip()
    if re.search(r"Timeout|SCREEN_SENSE|CYCLE|Calibration complete|BRAIN LOAD|File \"|Thread 0x|in <module>|Traceback", s):
        print(i, ":", s[:150])