# This Makefile is an alternative to installing Commander4 using `pip install -e . --no-build-isolation`
# The Makefile will not install Commander4 globally, but instead simply compile all support code
# Commander4 needs in order to function. Execute the Makefile by simply running `make`.
# You can then run Commander4 by executing the binary `bin/commander4` (with mpirun).

# --- Configuration ---
PYTHON := python
PIP := $(PYTHON) -m pip
CXX := g++
CC := gcc

# --- Compilation flags ---
# -O3: Max optimization
# -march=native: Optimize for the specific CPU (AVX, etc.) on this machine
# -ffast-math: Aggressive floating point math
# -fPIC: Position Independent Code (required for shared libraries)
# -shared: Create a shared library (.so)
CXXFLAGS := -O3 -Wall -fPIC -std=c++17 -shared -ffast-math -march=native

# Preprocessor definitions required by cmdr4_support.cc
# PKGNAME: The Python module name (used in PYBIND11_MODULE)
# PKGVERSION: Version string for the module
DEFINES := -DPKGNAME=cmdr4_support -DPKGVERSION=0.0.1

# Shell command asking pybind11 where its header files are.
PYTHON_INCLUDES := $(shell python3 -m pybind11 --includes)

# Shell command to figure out what the suffix of the exported module will end up looking like,
# typically something like ".cpython-313-x86_64-linux-gnu.so".
EXTENSION_SUFFIX := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Paths to the various codes
# The C++ directory.
SRC_CPP_DIR := src/lib_cpp
# The Python directory where the output .so files will be places.
DEST_DIR    := src/commander4
# ducc0 code directory.
DUCC_DIR    := external/ducc0/src

# Combine all includes
INCLUDES := -I$(SRC_CPP_DIR) -I$(DUCC_DIR) $(PYTHON_INCLUDES)

# --- Targets ---
# 1. Main Extension (cmdr4_support)
# This includes the core C++ logic + ducc0 helper files
TARGET_SUPPORT := $(DEST_DIR)/cmdr4_support$(EXTENSION_SUFFIX)
SOURCES_SUPPORT := $(SRC_CPP_DIR)/cmdr4_support.cc \
                   $(DUCC_DIR)/ducc0/infra/string_utils.cc \
                   $(DUCC_DIR)/ducc0/infra/threading.cc \
                   $(DUCC_DIR)/ducc0/infra/mav.cc

# 2. Mapmaker (mapmaker.so)
# Currently, the mapmaker uses Ctypes instead of Pybind11. We separately compile this code too.
TARGET_MAPMAKER := $(DEST_DIR)/mapmaker.so
SOURCES_MAPMAKER := $(SRC_CPP_DIR)/mapmaker.cpp

# --- Rules ---

.PHONY: all clean

# Define behavior of the `all` command
all: $(TARGET_SUPPORT) $(TARGET_MAPMAKER) stubs
	@echo "-------------------------------------------------------"
	@echo "Build complete."
	@echo "Binaries placed in: $(DEST_DIR)"
	@echo "You can now run:    python bin/commander4 -p params/..."
	@echo "-------------------------------------------------------"

# Build the Main Extension
$(TARGET_SUPPORT): $(SOURCES_SUPPORT)
	@echo "Compiling $(TARGET_SUPPORT)..."
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) $^ -o $@

# Build the Mapmaker
$(TARGET_MAPMAKER): $(SOURCES_MAPMAKER)
	@echo "Compiling $(TARGET_MAPMAKER)..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

stubs: $(TARGET_LIB)
	@echo "Generating Python stubs..."
	# We set PYTHONPATH so it finds the just-compiled library in src/
	PYTHONPATH=src python3 -m pybind11_stubgen commander4.cmdr4_support --output-dir src --root-suffix=""
	@echo "Stubs generated: src/commander4/cmdr4_support.pyi"

clean:
	@echo "Cleaning compiled binaries..."
	rm -f $(DEST_DIR)/*.so