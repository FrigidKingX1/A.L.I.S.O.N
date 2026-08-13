import re

pats = re.compile(r"Current thread|Thread 0x|File |CYCLE|Loading weights|in run_path")
for i, line in enumerate(open("auto_run2.log", encoding="utf-8", errors="replace"), 1):
    if pats.search(line):
        print(i, ":", line.rstrip()[:170])