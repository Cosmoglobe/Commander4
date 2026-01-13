from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Regenerate `.pyi` stubs for the compiled extension.

    Notes:
    - `pip install -e ".[stubs]"` only installs the dependency (`pybind11-stubgen`).
      It does not automatically run stub generation.
    - This command runs stubgen against the *installed* extension module.
    """

    # Ensure the extension is importable before trying to stubgen it.
    import commander4._cmdr4_backend  # noqa: F401

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "src"

    cmd = [
        sys.executable,
        "-m",
        "pybind11_stubgen",
        "commander4._cmdr4_backend",
        "--output-dir",
        str(out_dir),
        "--root-suffix",
        "",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
