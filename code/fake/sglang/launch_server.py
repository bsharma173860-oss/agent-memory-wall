import os, runpy, sys
sys.argv[0] = "sglang"
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../fakeserver.py"), run_name="__main__")
