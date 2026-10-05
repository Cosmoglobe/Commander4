"""Command-line tool: generate type stubs for the compiled backend."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Regenerate `.pyi` stubs for the compiled extension.

    Notes:
    - Stubs are normally generated during `pip install -e .`; this tool is for regenerating them
      by hand.
    - Uses `nanobind.stubgen`, which ships with the `nanobind` package. nanobind is only a build
      requirement, so it may be missing from the runtime environment: `pip install nanobind`.
    - This command runs stubgen against the *installed* extension module.
    """

    # Ensure the extension is importable before trying to stubgen it.
    import commander4._cmdr4_backend  # noqa: F401

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "src" / "commander4"

    try:
        import nanobind  # noqa: F401
    except ImportError:
        sys.exit("c4-generate-stubs needs nanobind in this environment: pip install nanobind")

    cmd = [
        sys.executable,
        "-m",
        "nanobind.stubgen",
        "-m",
        "commander4._cmdr4_backend",
        "-r",
        "-O",
        str(out_dir),
    ]

    # Start from an empty stub directory, so stubs of removed submodules do not linger.
    shutil.rmtree(out_dir / "_cmdr4_backend", ignore_errors=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
