PYTHON = python
PIP = $(PYTHON) -m pip
CXX = g++
CC = gcc
CXXFLAGS = -shared -O3 -Wall -fPIC -std=c++17

.PHONY: all
all: check-venv install compile

.PHONY: check-venv
check-venv:
ifeq ($(strip $(VIRTUAL_ENV)),)
	@echo "WARNING: No virtual Python environment detected."
	@echo "We recommend abort and set this up first by running 'python -m venv my_venv' and then 'source my_venv/bin/activate'."
	@echo -n "Do you want to proceed using the system Python? (y/N) "
	@read REPLY; \
	if [ "$$REPLY" != "y" ] && [ "$$REPLY" != "Y" ]; then \
		echo "Aborting make."; \
		exit 1; \
	fi
endif

.PHONY: install
install:
	@echo "--- Installing Python dependencies ---"
	$(PIP) install --upgrade pip
	CC=gcc CXX=g++ $(PIP) install --upgrade --no-cache-dir --no-binary ducc0 ducc0
	CC=gcc CXX=g++ $(PIP) install --upgrade --no-cache-dir --no-binary mpi4py mpi4py
	$(PIP) install -r requirements.txt
	@echo "\n--- Installing pybind11 support package ---"
	cd cmdr4_support && CC=gcc CXX=g++ $(PIP) install .
	@echo "\nAll Python dependencies installed."

.PHONY: compile
compile:
	@echo "\n--- Compiling C++ code for ctypes usage ---"
	cd src/cpp && $(CXX) $(CXXFLAGS) mapmaker.cpp -o mapmaker.so
	@echo "\nC++ code compiled."

.PHONY: clean
clean:
	@echo "\n--- Cleaning up ---"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf src/cpp/*.so
	@echo "\nCleanup complete."