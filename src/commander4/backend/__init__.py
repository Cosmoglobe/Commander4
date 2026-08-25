"""Single entry point for all compiled Commander4 code.

Two compiled artifacts live behind this package:
  - the pybind11 module ``commander4._cmdr4_backend`` (C++ sources in ``src/lib_cpp/``, built by
    CMake, type stubs in ``src/commander4/_cmdr4_backend/``), re-exported here so that callers
    write ``from commander4.backend import utils as cpp_utils``;
  - the ctypes shared library ``commander4/_libs/cmdr4_ctypes.so`` (C++ sources in
    ``src/lib_cpp/ctypes/``), loaded by ``ctypes_lib.load_cmdr4_ctypes_lib`` in this package.

The pybind11 module keeps its ``_cmdr4_backend`` name because it is fixed by the CMake ``PKGNAME``
variable and baked into the compiled ``.so``; renaming it requires rebuilding the package.
"""

from commander4._cmdr4_backend import *
