#!/usr/bin/env python3
"""Run the three experiments serially with explicit prerequisites and a time budget."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import yaml


class Experiment:
    def __init__(self, hours: float, poll_seconds: int):
        self.root = Path(__file__).resolve().parents[1]
        self.state_path = self.root / "docs/experiment_state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {"completed_phases": []}
        started = json.loads((self.root / "docs/bootstrap_manifest.json").read_text(encoding="utf-8"))["created_utc"]
        self.deadline = datetime.fromisoformat(started) + timedelta(hours=hours)
        self.poll_seconds = poll_seconds
        self.child = None
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self.stop)

    def stop(self, signum: int, _frame: object) -> None:
        if self.child is not None and self.child.poll() is None:
            # Only the child wrapper of this pipeline is signaled, never other jobs.
            self.child.send_signal(signum)
        self.update(status="interrupted")
        raise SystemExit(128 + signum)

    def update(self, **fields: object) -> None:
        self.state.update(fields)
        self.state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        self.state["deadline_utc"] = self.deadline.isoformat()
        temporary = self.state_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)
        print("PIPELINE_STATE " + json.dumps(fields, ensure_ascii=False), flush=True)

    def remaining(self) -> float:
        return (self.deadline - datetime.now(timezone.utc)).total_seconds()

    def check_budget(self) -> None:
        if self.remaining() <= 0:
            raise TimeoutError("The requested experiment budget elapsed; preserving all existing artifacts")

    def run(self, phase: str, arguments: list[str]) -> None:
        if phase in self.state["completed_phases"]:
            return
        self.check_budget()
        self.update(status="running", phase=phase)
        command = [sys.executable, str(self.root / "scripts/run.py"), "--log", phase, "--", *arguments]
        self.child = subprocess.Popen(command, cwd=self.root)
        try:
            self.child.wait(timeout=self.remaining())
        except subprocess.TimeoutExpired:
            self.child.send_signal(signal.SIGTERM)
            self.child.wait(timeout=60)
            raise TimeoutError(f"Budget elapsed while running {phase}")
        code = self.child.returncode
        self.child = None
        if code:
            raise RuntimeError(f"Phase {phase} exited {code}; inspect logs/{phase}.log")
        self.state["completed_phases"].append(phase)
        self.update(status="running", last_completed=phase)

    def script(self, phase: str, filename: str, *args: object) -> None:
        self.run(phase, [sys.executable, f"scripts/{filename}", *map(str, args)])

    def wait_resources(self, model: bool = False) -> None:
        last_reason = None
        last_report = 0.0
        while True:
            self.check_budget()
            reason = None
            if model and not (self.root / "models/qwen_image_2_1/snapshot_manifest.json").is_file():
                reason = "official_model_download"
            else:
                free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
                if free < 23000:
                    reason = "gpu_in_use_by_other_job"
                elif psutil.virtual_memory().available < 22 * 2**30:
                    reason = "host_ram_in_use_by_other_job"
            if reason is None:
                return
            if reason != last_reason or time.monotonic() - last_report >= 600:
                self.update(status="waiting", waiting_for=reason, remaining_hours=round(self.remaining() / 3600, 2))
                last_reason, last_report = reason, time.monotonic()
            time.sleep(min(self.poll_seconds, max(1, self.remaining())))

    def derived_config(self, source: str, target: str, mutate) -> Path:
        path = self.root / target
        config = yaml.safe_load((self.root / source).read_text(encoding="utf-8"))
        mutate(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path.relative_to(self.root)

    def execute(self) -> None:
        self.wait_resources(model=True)
        self.script("audit_project", "audit_project.py", "--verify-hashes")
        self.script("test_software", "test_software.py")
        self.script("test_quantized_lora", "test_quantized_lora.py")
        self.wait_resources()
        # Keep this retry separate from the earlier failed GPU probes: the
        # timing budget must be based only on a completed, contiguous run.
        base_probe_dir = "test_artifacts/gpu_smoke/base_probe_gc_stable"
        self.script("evaluate_base_probe", "evaluate.py", "--stage", "base", "--split", "development", "--limit", 2,
                    "--seeds", 43, "--output", base_probe_dir)
        self.wait_resources()
        self.script("train_sft_smoke", "train_stage.py", "configs/sft_smoke.yaml", "--probe")
        probe = json.loads((self.root / base_probe_dir / "metrics.json").read_text(encoding="utf-8"))
        cases = json.loads((self.root / base_probe_dir / "cases.json").read_text(encoding="utf-8"))
        sft_smoke = json.loads((self.root / "test_artifacts/gpu_smoke/sft_smoke_rgba/stage_summary.json").read_text(encoding="utf-8"))
        if "budget_plan" not in self.state:
            generation = probe["mean_generation_seconds"]
            encoding = sum(row["encoding_seconds"] for row in cases) / len(cases)
            per_example = sft_smoke["optimization_seconds"] / 8
            epoch_seconds = 512 * per_example * 1.4
            preparation_seconds = 512 * encoding * 1.4
            initialization = sft_smoke["initialization_seconds"]
            # Choose sample count and epochs using timing only, never test quality.
            for evaluation_count in (64, 32, 16):
                eval_seconds = evaluation_count * (encoding + 2 * generation) * 1.6 + 2 * initialization
                available = self.remaining() - 3 * eval_seconds - preparation_seconds - 4 * initialization
                if available > epoch_seconds * 1.6:
                    break
            else:
                raise TimeoutError("Insufficient measured budget for three evaluations and nontrivial training")
            epochs = max(1, min(3, int(available * 0.30 / epoch_seconds)))
            plan = {"evaluation_cases": evaluation_count, "evaluation_seeds": [43, 20261005], "sft_epochs": epochs,
                    "sft_examples": 512, "estimated_epoch_seconds": epoch_seconds,
                    "estimated_evaluation_seconds_per_stage": eval_seconds,
                    "selection_basis": "development/smoke execution time only; fixed final checkpoint; no test selection"}
            self.update(budget_plan=plan)
        plan = self.state["budget_plan"]
        self.wait_resources()
        self.script("evaluate_base", "evaluate.py", "--stage", "base", "--limit", plan["evaluation_cases"])
        sft_config = self.derived_config("configs/sft.yaml", "configs/sft_budget.yaml",
                                         lambda config: config["train"].update(max_epochs=plan["sft_epochs"]))
        self.wait_resources()
        self.script("train_sft", "train_stage.py", sft_config)
        self.wait_resources()
        self.script("evaluate_sft", "evaluate.py", "--stage", "sft", "--limit", plan["evaluation_cases"])

        def configure_grpo_smoke(config: dict) -> None:
            config["model"]["resume_path"] = "runs/sft/export"
            # Keep this smoke recipe separate from both failed prior attempts.
            # This remains genuine Flow-GRPO: group-relative rewards,
            # rollout/replay likelihood ratios, and an SFT reference KL are
            # all retained.
            config["log"]["run_name"] = "grpo_smoke_nocache16_medium"
            config["train"].update(
                use_kv_cache=False,
                num_inference_steps=16,
                group_size=2,
                gradient_accumulation_steps=30,
            )
            config["scheduler"].update(num_sde_steps=15, sde_steps=list(range(15)))
        smoke_config = self.derived_config(
            "configs/grpo_smoke.yaml",
            "test_artifacts/gpu_smoke/grpo_on_sft_nocache16_medium.yaml",
            configure_grpo_smoke,
        )
        self.wait_resources()
        self.script("train_grpo_smoke_nocache16_medium", "train_stage.py", smoke_config, "--probe")
        grpo_smoke = json.loads((self.root / "test_artifacts/gpu_smoke/grpo_smoke_nocache16_medium/stage_summary.json").read_text(encoding="utf-8"))
        if "grpo_cycles" not in plan:
            # The formal GRPO recipe now matches this smoke recipe in group
            # size and trajectory length, so retain only a timing cushion.
            cycle_seconds = grpo_smoke["optimization_seconds"] * 1.4
            reserve = plan["estimated_evaluation_seconds_per_stage"] + 2 * grpo_smoke["initialization_seconds"] + 3600
            cycles = min(100, int((self.remaining() - reserve) / cycle_seconds))
            if cycles < 8:
                raise TimeoutError("Measured budget cannot support at least eight GRPO cycles and the final evaluation")
            plan.update(grpo_cycles=cycles, estimated_grpo_cycle_seconds=cycle_seconds)
            self.update(budget_plan=plan)
        def configure_grpo_budget(config: dict) -> None:
            config["train"].update(
                max_epochs=plan["grpo_cycles"],
                use_kv_cache=False,
                num_inference_steps=16,
                group_size=2,
                gradient_accumulation_steps=30,
            )
            config["scheduler"].update(num_sde_steps=15, sde_steps=list(range(15)))
        grpo_config = self.derived_config("configs/grpo.yaml", "configs/grpo_budget.yaml", configure_grpo_budget)
        self.wait_resources()
        self.script("train_grpo", "train_stage.py", grpo_config)
        self.wait_resources()
        self.script("evaluate_grpo", "evaluate.py", "--stage", "grpo", "--limit", plan["evaluation_cases"])
        self.script("compare_experiments", "compare_experiments.py")
        self.script("audit_final", "audit_project.py", "--verify-hashes")
        self.update(status="complete", phase="three_group_comparison", waiting_for=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-hours", type=float, default=72)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if not 0 < args.budget_hours <= 72 or not 10 <= args.poll_seconds <= 600:
        parser.error("Budget must be at most 72 hours; polling must be 10..600 seconds")
    experiment = Experiment(args.budget_hours, args.poll_seconds)
    if experiment.state.get("status") == "complete":
        print("PIPELINE_ALREADY_COMPLETE", flush=True)
        return
    if experiment.state.get("status") == "failed":
        # Keep the original failed-run log for diagnosis, but make the live
        # state unambiguous once an operator deliberately resumes the pipeline.
        experiment.update(status="resuming", error=None)
    try:
        experiment.execute()
    except Exception as error:
        experiment.update(status="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
