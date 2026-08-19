from __future__ import annotations

import ctypes as ct
from pathlib import Path


def load_cmdr4_ctypes_lib() -> ct.CDLL:
    """Load the packaged Commander4 ctypes shared library (Linux-only).

    Notes:
    - In editable installs, Python sources may come from the repo (src-layout)
      while compiled artifacts are installed into site-packages. The package
      __path__ typically contains both locations, so search all of them.
    - The library is installed as commander4/_libs/cmdr4_ctypes.so.
    """

    import commander4

    pkg_dirs = [Path(p).resolve() for p in getattr(commander4, "__path__", [])]
    if not pkg_dirs:
        pkg_dirs = [Path(commander4.__file__).resolve().parent]

    rel_path = Path("_libs") / "cmdr4_ctypes.so"

    for pkg_dir in pkg_dirs:
        path = pkg_dir / rel_path
        if path.exists():
            return ct.CDLL(str(path))

    raise FileNotFoundError(
        "Could not locate packaged cmdr4_ctypes shared library. "
        f"Expected to find {rel_path} under one of: {', '.join(str(p) for p in pkg_dirs)}"
    )
