#!/usr/bin/env python3
"""Pin the actual runtime dependency closure without machine-specific file URLs."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT_PACKAGES = ["torch", "torchvision", "accelerate", "peft", "transformers", "bitsandbytes", "datasets",
                 "av", "rapidocr", "onnxruntime", "pytest", "sentencepiece", "protobuf", "ftfy", "imageio",
                 "imageio-ffmpeg", "numpy", "pillow", "einops", "pydantic", "pyyaml", "requests", "diffusers", "flow-factory"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # This experiment uses bitsandbytes only; a stale optional backend breaks Diffusers imports.
    try:
        metadata.version("torchao")
    except metadata.PackageNotFoundError:
        pass
    else:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "torchao"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub==1.32.0", "transformers==5.18.0",
                    "peft==0.21.2", "rapidocr==3.9.2", "ftfy", "imageio[ffmpeg]>=2.37.2",
                    "protobuf>=6.33.2", "pydantic>=2.8", "einops>=0.8"], check=True)
    versions = {}
    pending = list(ROOT_PACKAGES)
    while pending:
        name = canonicalize_name(pending.pop())
        if name in versions:
            continue
        distribution = metadata.distribution(name)
        versions[name] = distribution.version
        for dependency in distribution.requires or []:
            requirement = Requirement(dependency)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                installed = metadata.version(requirement.name)
                if requirement.specifier and installed not in requirement.specifier:
                    raise RuntimeError(f"Unsatisfied runtime dependency: {name} needs {requirement}; installed {installed}")
                pending.append(requirement.name)
    # Both backends are installed from the ordinary pinned source trees, not file:// URLs.
    lines = [f"{name}=={version}" for name, version in sorted(versions.items()) if name not in {"flow-factory", "diffusers"}]
    (root / "requirements.lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "docs/runtime_versions.json").write_text(json.dumps({"python": sys.version, "packages": versions}, indent=2) + "\n", encoding="utf-8")
    print("RUNTIME_ENVIRONMENT_FINALIZED " + json.dumps(versions), flush=True)


if __name__ == "__main__":
    main()
