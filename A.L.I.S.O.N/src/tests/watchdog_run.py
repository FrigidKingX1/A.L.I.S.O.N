import faulthandler, runpy, sys

faulthandler.dump_traceback_later(300, exit=True)
sys.argv = ["alison_core.py", "--auto", "--cycles", "3"]
runpy.run_path("alison_core.py", run_name="__main__")