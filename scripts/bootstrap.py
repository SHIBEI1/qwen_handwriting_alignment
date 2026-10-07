#!/usr/bin/env python3
"""Materialize pinned source trees and build an external, independent environment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


REPOSITORIES = {
    "flow_factory": ("https://github.com/X-GenGroup/Flow-Factory.git", "f847319079255e8506604bb3aec261fc27708935"),
    "diffusers": ("https://github.com/huggingface/diffusers.git", "80c7ed262aeffbeb43ef13ae04baeb9b84515a69"),
}


def run(command: list[str], cwd: Path | None = None) -> None:
    print(f"COMMAND {command!r}", flush=True)
    subprocess.run(command, check=True, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda", required=True)
    parser.add_argument("--clone-env", required=True)
    parser.add_argument("--env-prefix", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for folder in ("models", "data", "configs", "src/handwriting_align", "test_artifacts", "docs", "vendor", "runs"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    state = {"created_utc": datetime.now(timezone.utc).isoformat(), "repositories": {}}
    for name, (url, revision) in REPOSITORIES.items():
        target = root / "vendor" / name
        if not target.exists():
            run(["git", "clone", "--no-recurse-submodules", "--filter=blob:none", url, str(target)])
            run(["git", "checkout", "--detach", revision], cwd=target)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Unexpected {name} revision: {actual}; required {revision}")
        state["repositories"][name] = {"url": url, "revision": actual}
    for intent in ("develop", "new-reward"):
        run([sys.executable, "scripts/agent_scope.py", "--base", "origin/main", "--intent", intent], cwd=root / "vendor/flow_factory")
    prefix = Path(args.env_prefix).resolve()
    if root == prefix or root in prefix.parents:
        raise ValueError("The environment must live outside this project")
    if not prefix.exists():
        run([args.conda, "create", "--yes", "--copy", "--prefix", str(prefix), "--clone", args.clone_env])
    python = str(prefix / "bin/python")
    run([python, "-m", "pip", "install", "huggingface-hub==1.32.0", "transformers==5.18.0", "peft==0.21.2", "datasets>=3.3,<5", "av>=17", "rapidocr==3.9.2", "onnxruntime", "pytest", "sentencepiece", "protobuf>=6.33.2", "ftfy", "imageio[ffmpeg]>=2.37.2", "pydantic>=2.8", "einops>=0.8"])
    run([python, "-m", "pip", "install", "--no-deps", str(root / "vendor/diffusers")])
    run([python, "-m", "pip", "install", "--no-deps", str(root / "vendor/flow_factory")])
    state["environment_prefix_at_build"] = str(prefix)
    state["python_version"] = subprocess.check_output([python, "--version"], text=True).strip()
    (root / "docs/bootstrap_manifest.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run([python, str(root / "scripts/normalize_resources.py")])
    run([python, str(root / "scripts/finalize_environment.py")])
    print("BOOTSTRAP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
