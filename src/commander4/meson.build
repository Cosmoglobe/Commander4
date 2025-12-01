# This is the Meson file that builds both the Python and C++ code.

# --- 1. Python Auto-Discovery ---
# By default, in Meson, you need to list every single Python file you want to include in your build.
# We instead run a tiny Python script to list every .py file in our directory tree.
# This means we do not have to change this build script if adding more Python files.
glob_script = '''
import glob
import os
import sys

# 1. Enter the correct directory (passed as argument)
target_dir = sys.argv[1]
os.chdir(target_dir)

# 2. Find all Python files
files = glob.glob("**/*.py", recursive=True)

# 3. Print each file on a new line
for f in files:
    print(f)
'''

# Execute the script, passing the current directory as an argument
py_files_raw = run_command(py, '-c', glob_script, meson.current_source_dir(), 
  check: true
).stdout().strip()

# Split the output string into a list
if py_files_raw == ''
  python_sources = []
else
  python_sources = py_files_raw.split('\n')
endif

# --- Submodule Check ---
# ducc0 is included as a git submodule. Check for a known file inside the submodule to see if
# it is present. If not, throw an error, as Meson cannot pull the submodule itself.
ducc0_check = run_command(py, '-c', 
  'import os; print(os.path.exists("../../external/ducc0/src/ducc0/infra/mav.cc"))', 
  check: true
).stdout().strip()

if ducc0_check != 'True'
  error('The "ducc0" submodule is missing! Please run: git submodule update --init --recursive')
endif

# Install every Python file found
# preserve_path: true ensures that e.g. 'utils/math.py' is installed as 'utils/math.py'
py.install_sources(
  python_sources,
  subdir : 'commander4',
  preserve_path : true
)

# 2. C++ Configuration
cc = meson.get_compiler('cpp')

# --- Optimization Flags ---
# Meson 'buildtype=release' gives -O3. We add the rest manually.
extra_args = [
  '-march=native',  # Enable CPU-specific optimization (e.g. AVX2 or AVX512)
  '-ffast-math', 
  '-Wfatal-errors', 
  '-Wfloat-conversion',
  '-DPKGNAME=cmdr4_support',  # Needs to match name of pybind11 C++ module.
  '-DPKGVERSION=0.1.0'
]

host_system = host_machine.system()
host_cpu = host_machine.cpu_family()

# --- Include Directories ---
# Relative paths to included directories.
incdir = include_directories('.', '../lib_cpp', '../../external/ducc0/src')

# --- Source Files ---
cpp_sources = [
  '../lib_cpp/cmdr4_support.cc',
  '../../external/ducc0/src/ducc0/infra/string_utils.cc',
  '../../external/ducc0/src/ducc0/infra/threading.cc',
  '../../external/ducc0/src/ducc0/infra/mav.cc',
  ]

# --- Build the Module ---
py.extension_module(
  'cmdr4_support',
  cpp_sources,
  include_directories : [incdir, include_directories(pybind11_inc)],
  dependencies : [py_dep],
  cpp_args : extra_args,
  install : true,
  subdir : 'commander4'
)