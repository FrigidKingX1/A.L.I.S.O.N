lines = open("auto_run2.log", encoding="utf-8", errors="replace").read().splitlines()
threads = [
    (i, l) for i, l in enumerate(lines)
    if "Thread 0x" in l and "most recent call first" in l
]
main = None
for idx, (i, l) in enumerate(threads):
    blk = lines[i:i + 8]
    if any("3320" in b for b in blk):
        main = blk
        break
if main:
    print("\n".join(main))
else:
    print("no main-thread block found; total threads:", len(threads))