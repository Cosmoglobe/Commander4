import sys, os, subprocess

# 1. Detect CPU architecture
try:
    cmd = "g++ -march=native -Q --help=target | grep -m1 'march=' | cut -d= -f2"
    arch = subprocess.check_output(cmd, shell=True).decode().strip()
except Exception:
    arch = "generic"

# 2. Find the correct backend folder for our CPU in ../backends/<cpu_architecture>/
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib_path = os.path.join(basedir, "backends", arch)

if not os.path.isdir(lib_path):
    raise ImportError(f"Binary not found in {lib_path}. Run 'make'.")

# 3. Import
# We insert the folder into sys.path, import the module, then clean up.
sys.path.insert(0, lib_path)
try:
    from _cmdr4_backend import *
finally:
    del sys.path[0]