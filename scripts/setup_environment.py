#!/usr/bin/env python3
"""Rebuild the external environment from this project's ordinary files only."""

import argparse
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda", required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prefix = args.prefix.resolve()
    if root == prefix or root in prefix.parents:
        parser.error("Choose an environment outside the project")
    if not prefix.exists():
        subprocess.run([args.conda, "create", "--yes", "--copy", "--prefix", str(prefix), "python=3.12", "pip"], check=True)
    python = prefix / "bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"Expected a Linux conda Python at {python}")
    subprocess.run([str(python), "-m", "pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cu130",
                    "--requirement", str(root / "requirements.lock.txt")], check=True)
    for name in ("diffusers", "flow_factory"):
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", str(root / "vendor" / name)], check=True)
    print(f"PORTABLE_ENVIRONMENT_SETUP_COMPLETE {prefix}", flush=True)


if __name__ == "__main__":
    main()
