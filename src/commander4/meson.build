# Meson build file located at src/commander4/meson.build

# 1. Glob for all Python files relative to this directory
glob_script = '''
import glob
import os
import sys

# Change directory to the location of this meson.build file
os.chdir(sys.argv[1])

# Find all .py files recursively
files = glob.glob("**/*.py", recursive=True) + glob.glob("**/*.pyi", recursive=True)

# Print them line by line
for f in files:
    print(f)
'''

# 2. Run the script
py_files = run_command(py, '-c', glob_script, meson.current_source_dir(), check: true).stdout().strip().split('\n')

# 3. Install the sources
py.install_sources(
  py_files,
  subdir : 'commander4',
  preserve_path : true  # Ensures that e.g. 'output/log.py' stays inside 'output/'.
)