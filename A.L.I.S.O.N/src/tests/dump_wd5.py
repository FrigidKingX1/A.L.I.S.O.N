lines = open("watchdog5.log", encoding="utf-16", errors="replace").read().splitlines()
for i in range(200, min(len(lines), 300)):
    print(i, ":", lines[i].rstrip()[:160])
