import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
text = open("auto_run2.log", encoding="utf-8", errors="replace").read()
i = text.find("Timeout")
if i < 0:
    i = text.find("Timeout")
print(text[-6000:])