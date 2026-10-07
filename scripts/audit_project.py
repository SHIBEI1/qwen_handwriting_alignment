#!/usr/bin/env python3
"""Fail on filesystem links, external dataset assets, or mutable source revisions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    links, hardlinks, files = [], [], 0
    for folder, directories, names in os.walk(root, followlinks=False):
        for name in [*directories, *names]:
            path = Path(folder) / name
            if path.is_symlink():
                links.append(str(path.relative_to(root)))
            elif path.is_file():
                files += 1
                if path.stat().st_nlink > 1:
                    hardlinks.append(str(path.relative_to(root)))
    snapshots = {}
    for name, revision in (("flow_factory", "f847319079255e8506604bb3aec261fc27708935"),
                           ("diffusers", "80c7ed262aeffbeb43ef13ae04baeb9b84515a69")):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root / "vendor" / name, text=True).strip()
        assert actual == revision
        snapshots[name] = actual
    data = root / "data/native_subset"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (data / item["path"]).resolve()
        assert root in path.parents and path.is_file()
        assert path.stat().st_size == item["bytes"]
        if args.verify_hashes:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == item["sha256"], f"Dataset hash mismatch: {item['path']}"
    model_manifest = root / "models/qwen_image_2_1/snapshot_manifest.json"
    if model_manifest.is_file():
        snapshot = json.loads(model_manifest.read_text(encoding="utf-8"))
        assert snapshot["model_id"] == "Qwen/Qwen-Image-2.1"
        assert snapshot["revision"] == "d26bb61231c349cf6b7896fa83353113880e1ba3"
        for item in snapshot["files"]:
            path = (model_manifest.parent / item["path"]).resolve()
            assert model_manifest.parent in path.parents and path.is_file()
            assert path.stat().st_size == item["bytes"]
            if args.verify_hashes:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                assert digest.hexdigest() == item["sha256"], f"Model hash mismatch: {item['path']}"
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "regular_file_count": files,
              "symlinks": links, "hardlinks": hardlinks, "pinned_repositories": snapshots,
              "dataset_assets_inside_project": True,
              "official_model_download_complete": model_manifest.exists(), "hashes_verified": args.verify_hashes}
    (root / "docs/independence_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PROJECT_AUDIT " + json.dumps(report), flush=True)
    if links or hardlinks:
        raise RuntimeError("The project contains filesystem links")


if __name__ == "__main__":
    main()
