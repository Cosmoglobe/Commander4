"""Single entry point for all compiled Commander4 code.

All compiled code is the nanobind module ``commander4._cmdr4_backend`` (C++ sources in
``src/lib_cpp/``, one ``*_pymod.cc`` file per submodule, built by CMake, type stubs in
``src/commander4/_cmdr4_backend/``). Its submodules (``utils``, ``mapmaker``, ``compsep``) are
re-exported here so that callers write e.g. ``from commander4.backend import mapmaker as
cpp_mapmaker``.

The nanobind module keeps its ``_cmdr4_backend`` name because it is fixed by the CMake ``PKGNAME``
variable and baked into the compiled ``.so``; renaming it requires rebuilding the package.
"""

from commander4._cmdr4_backend import *
