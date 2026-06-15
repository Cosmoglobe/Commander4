#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$script_dir/fast_ang2pix.cpp"
out="$script_dir/fast_ang2pix_ctypes.so"
if [[ -n "${CXX:-}" ]]; then
	cxx="$CXX"
elif command -v icpx >/dev/null 2>&1; then
	cxx="$(command -v icpx)"
else
	cxx="g++"
fi

compiler_name="$(basename "$cxx")"

if [[ "$compiler_name" == "icpx" ]]; then
	report_file="$script_dir/fast_ang2pix.optrpt"
	rm -f "$report_file"
	cmd=(
		"$cxx"
		-std=c++17
		-O3
		-march=native
		-ffast-math
		-fno-math-errno
		-fno-trapping-math
		-qopenmp
		-fPIC
		-shared
		-Wall
		-Wextra
		-Wpedantic
		-qopt-report=max
		-qopt-report-phase=vec
		-o
		"$out"
		"$src"
		-lm
		-v
	)
else
	report_file="$script_dir/fast_ang2pix.vec.txt"
	rm -f "$report_file"
	cmd=(
		"$cxx"
		-std=c++17
		-O3
		-Ofast
		-march=native
		-mtune=native
		-ffast-math
		-fno-math-errno
		-fno-trapping-math
		-fopenmp
		-fopenmp-simd
		-fPIC
		-shared
		-Wall
		-Wextra
		-Wpedantic
		"-fopt-info-vec-all=$report_file"
		-o
		"$out"
		"$src"
		-lm
		-v
	)
fi

printf 'Building %s -> %s\n' "$src" "$out"
printf 'Compiler: %s\n' "$cxx"
printf 'Vectorization report:\n  %s\n' "$report_file"

"${cmd[@]}"