#!/usr/bin/env python3
"""Replace verified in-project file links with independent ordinary files."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    replacements = []
    # Validate the complete operation before replacing any path.
    for path in sorted(root.rglob("*")):
        if not path.is_symlink() and not (path.is_file() and path.stat().st_nlink > 1):
            continue
        resolved = path.resolve(strict=True)
        if root not in resolved.parents or not resolved.is_file():
            raise RuntimeError(f"Refusing an external or directory link: {path} -> {resolved}")
        replacements.append((path, resolved, path.is_symlink()))
    records = []
    for path, resolved, symbolic in replacements:
        payload = resolved.read_bytes()
        mode = stat.S_IMODE(resolved.stat().st_mode)
        descriptor, temporary = tempfile.mkstemp(prefix=".materialize-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        record = {"path": str(path.relative_to(root)), "original_target": str(resolved.relative_to(root)),
                  "link_type": "symbolic" if symbolic else "hard", "sha256": hashlib.sha256(payload).hexdigest()}
        records.append(record)
        print("MATERIALIZED " + json.dumps(record), flush=True)
    report_path = root / "docs/vendor_materialization.json"
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"operations": []}
    previous["operations"].append({"utc": datetime.now(timezone.utc).isoformat(), "files": records})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    print(f"MATERIALIZATION_COMPLETE replaced={len(records)}", flush=True)


if __name__ == "__main__":
    main()
