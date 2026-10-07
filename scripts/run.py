#!/usr/bin/env python3
"""Run a project command with one timestamped, replace-on-run program log."""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.log):
        parser.error("--log must be a simple program name")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    root = Path(__file__).resolve().parents[1]
    for folder in ("logs", "cache", "cache/tmp", "cache/huggingface", "cache/torch"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    with (root / "logs" / f"{args.log}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / "logs" / f"{args.log}.log").open("w", encoding="utf-8", buffering=1) as log:
            def emit(message: str) -> None:
                stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
                line = f"[{stamp}] {message.rstrip()}"
                log.write(line + "\n")
                print(line, flush=True)

            env = os.environ.copy()
            env.update({
                "PYTHONUNBUFFERED": "1",
                "HANDWRITING_PROJECT_ROOT": str(root),
                "HF_HOME": str(root / "cache/huggingface"),
                "TORCH_HOME": str(root / "cache/torch"),
                "TMPDIR": str(root / "cache/tmp"),
                "TOKENIZERS_PARALLELISM": "false",
                # GRPO has a near-capacity but variable CUDA allocation pattern.
                # Expandable segments reduce allocator fragmentation without
                # changing tensors, weights, or optimization semantics.
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root / "vendor/flow_factory/src"), str(root / "vendor/diffusers/src"), env.get("PYTHONPATH", ""))),
            })
            lib = Path(sys.prefix) / "lib"
            if (lib / "libstdc++.so.6").is_file():
                env["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            emit(f"START program={args.log} cwd={root} command={command!r}")
            process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                       errors="replace", bufsize=1, start_new_session=True)

            def stop(signum: int, _frame: object) -> None:
                emit(f"SIGNAL {signum}; forwarding to command process group {process.pid}")
                if process.poll() is None:
                    os.killpg(process.pid, signum)

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            assert process.stdout is not None
            artifact_directory = None
            for line in process.stdout:
                emit(line)
                if line.startswith("TEST_ARTIFACT_DIR="):
                    candidate = Path(line.partition("=")[2].strip()).resolve()
                    if root / "test_artifacts" in candidate.parents and candidate.is_dir():
                        artifact_directory = candidate
            result = process.wait()
            emit(f"END program={args.log} exit_code={result}")
            if result == 0 and artifact_directory is not None:
                log.flush()
                shutil.copyfile(root / "logs" / f"{args.log}.log", artifact_directory / f"{args.log}.log")
            return result


if __name__ == "__main__":
    raise SystemExit(main())
