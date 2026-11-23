"""Runs the paper's code/agent_ctrl.py unchanged, except that it does not also copy its results into the root
directory (/), as it did on the rented GPU boxes of the study: the kit keeps everything in its --out-dir."""
import os, runpy, shutil, sys

CODE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
_copy = shutil.copy


def copy(src, dst, **kw):
    if os.path.dirname(os.path.abspath(dst)) == os.path.abspath(os.sep):
        return dst   # not into /
    return _copy(src, dst, **kw)


shutil.copy = copy
sys.path.insert(0, CODE)
sys.argv[0] = os.path.join(CODE, "agent_ctrl.py")
runpy.run_path(sys.argv[0], run_name="__main__")
