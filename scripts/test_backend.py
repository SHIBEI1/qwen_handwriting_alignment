#!/usr/bin/env python3
"""Run pinned-backend CPU contracts without downloads or pretrained quality claims."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "test_artifacts" / ("backend_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    commands = [
        ("diffusers_skills_inventory", [str(Path(sys.executable).parent / "diffusers-cli"), "skills", "list"], root / "vendor/diffusers"),
        ("diffusers_agent_guide_check", [sys.executable, "utils/check_ai.py"], root / "vendor/diffusers"),
        ("flow_factory_cpu_contracts", [sys.executable, "-m", "pytest", "-q", "-o", f"cache_dir={output / 'pytest_cache'}",
            f"--basetemp={output / 'pytest_temp'}",
            f"--junitxml={output / 'pytest.xml'}", "tests/models/test_qwen_image_21.py",
            "tests/scheduler/test_ode_dtype_consistency.py", "tests/scheduler/test_schedule_bit_parity.py",
            "tests/scheduler/test_scheduler_group.py", "tests/trainers/test_offline_flow_matching.py",
            "tests/trainers/test_common_primitives.py", "tests/trainers/test_coupled_trainer_parity.py"], root / "vendor/flow_factory"),
    ]
    results = []
    for name, command, directory in commands:
        print(f"BACKEND_CHECK {name} cwd={directory} command={command}", flush=True)
        with (output / f"{name}.txt").open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=directory, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
        code = process.wait()
        results.append({"check": name, "exit_code": code})
    passed = all(row["exit_code"] == 0 for row in results)
    report = {"checks_passed": passed, "checks": results,
              "scope": "CPU fake fixtures and scheduler/trainer mathematics; not pretrained GPU validation"}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("BACKEND_TEST_RESULTS " + json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError(f"A backend check failed; inspect {output}")
    print(f"TEST_ARTIFACT_DIR={output}", flush=True)


if __name__ == "__main__":
    main()
