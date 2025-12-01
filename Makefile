# This Makefile is an alternative to installing Commander4 using `pip install -e . --no-build-isolation`
# The Makefile will not install Commander4 globally, but instead simply compile all support code
# Commander4 needs in order to function. Execute the Makefile by simply running `make`.
# You can then run Commander4 by executing the binary `bin/commander4` (with mpirun).

# --- Configuration ---
PYTHON := python
PIP := $(PYTHON) -m pip
CXX := g++
CC := gcc
MODULE_NAME := _cmdr4_backend

# --- Architecture Detection ---
# We use 'shell' to get the CPU archtecture. If it fails or returns empty, default to 'generic'.
CPU_ARCH := $(shell $(CXX) -march=native -Q --help=target | grep -m1 "march=" | cut -d= -f2 | xargs)
ifeq ($(CPU_ARCH),)
    CPU_ARCH := generic
endif

# --- Paths ---
# Backends: Hidden folder where compiled binaries go
BACKEND_DIR := src/commander4/backends/$(CPU_ARCH)
# Support: The public Python package
SUPPORT_DIR := src/commander4/cmdr4_support
# The C++ directory.
SRC_CPP_DIR := src/lib_cpp
# The Python directory where the output .so files will be places.
DEST_DIR    := src/commander4
# ducc0 code directory.
DUCC_DIR    := external/ducc0/src

# --- Compilation flags ---
# -O3: Max optimization
# -march=native: Optimize for the specific CPU (AVX, etc.) on this machine
# -ffast-math: Aggressive floating point math
# -shared: Create a shared library (.so)
# -fPIC: Position Independent Code (required for shared libraries)
CXXFLAGS := -O3 -march=native -ffast-math -shared -fPIC -Wall -std=c++17

# Preprocessor definitions required by cmdr4_support.cc
# PKGNAME: The Python module name (used in PYBIND11_MODULE)
# PKGVERSION: Version string for the module
DEFINES  := -DPKGNAME=$(MODULE_NAME) -DPKGVERSION=0.1.0

# Shell command asking pybind11 where its header files are.
PYTHON_INCLUDES := $(shell python3 -m pybind11 --includes)

# Shell command to figure out what the suffix of the exported module will end up looking like,
# typically something like ".cpython-313-x86_64-linux-gnu.so".
EXTENSION_SUFFIX := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Combine all includes
INCLUDES := -I$(SRC_CPP_DIR) -I$(DUCC_DIR) $(PYTHON_INCLUDES)

# --- Targets ---
# 1. Main Extension (cmdr4_support)
# This includes the core C++ logic + ducc0 helper files
TARGET_SUPPORT := $(BACKEND_DIR)/$(MODULE_NAME)$(EXTENSION_SUFFIX)
SOURCES_SUPPORT := $(SRC_CPP_DIR)/cmdr4_support.cc \
                   $(DUCC_DIR)/ducc0/infra/string_utils.cc \
                   $(DUCC_DIR)/ducc0/infra/threading.cc \
                   $(DUCC_DIR)/ducc0/infra/mav.cc

# 2. Mapmaker (mapmaker.so)
# Currently, the mapmaker uses Ctypes instead of Pybind11. We separately compile this code too.
TARGET_MAPMAKER := $(DEST_DIR)/mapmaker.so
SOURCES_MAPMAKER := $(SRC_CPP_DIR)/mapmaker.cpp

# --- Rules ---

# Define behavior of the `all` command
.PHONY: all
all: check-submodules $(TARGET_SUPPORT) $(TARGET_MAPMAKER) stubs
	@echo "-------------------------------------------------------"
	@echo "Build complete."
	@echo "Binaries placed in: $(DEST_DIR)"
	@echo "You can now run:    python bin/commander4 -p params/..."
	@echo "-------------------------------------------------------"

# ducc0 is included as a git submodule. Check if ducc0 dir contains files. If not, try to init it.
.PHONY: check-submodules
check-submodules:
	@if [ -z "$$(ls -A external/ducc0)" ]; then \
		echo "Submodule 'ducc0' is empty. Initializing..."; \
		git submodule update --init --recursive || { echo "Git submodule init failed! Please run 'git submodule update --init --recursive' manually."; exit 1; }; \
	fi

# Build the Main Extension
$(TARGET_SUPPORT): $(SOURCES_SUPPORT)
	@mkdir -p $(BACKEND_DIR)
	@echo "Compiling $(TARGET_SUPPORT) for $(CPU_ARCH)..."
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) $^ -o $@

# Build the Mapmaker
$(TARGET_MAPMAKER): $(SOURCES_MAPMAKER)
	@echo "Compiling $(TARGET_MAPMAKER)..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# Generating stubs
# To allow linters etc to be able to understand the C++ backend we auto-generate "stubs", which are
# essentially Python header files with docstrings etc.
.PHONY: stubs
stubs: $(TARGET_BIN)
	@echo "Generating stubs..."
	PYTHONPATH=src $(PYTHON) -m pybind11_stubgen commander4.cmdr4_support --output-dir src --root-suffix=""
	@echo "Stubs generated in $(SUPPORT_DIR)"

.PHONY: clean
clean:
	@echo "Cleaning compiled binaries..."
	rm -rf src/commander4/backends