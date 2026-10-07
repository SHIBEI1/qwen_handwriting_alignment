#!/usr/bin/env python3
"""Download an immutable official model snapshot as verified ordinary files."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--revision", default="d26bb61231c349cf6b7896fa83353113880e1ba3")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--metadata-file", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / "models/qwen_image_2_1"
    target.mkdir(parents=True, exist_ok=True)
    endpoint = args.endpoint.rstrip("/")
    url = f"{endpoint}/api/models/{args.model_id}/revision/{args.revision}?blobs=true"
    print(f"FETCH immutable official revision metadata via {endpoint}", flush=True)
    if args.metadata_file:
        meta = json.loads(args.metadata_file.read_text(encoding="utf-8-sig"))
    else:
        with urllib.request.urlopen(url, timeout=30) as response:
            meta = json.load(response)
    if meta["sha"] != args.revision:
        raise RuntimeError("Official snapshot revision mismatch")
    files = [item for item in meta["siblings"] if not item["rfilename"].startswith(("assets/", "."))]
    total = sum(item.get("size", 0) for item in files)
    print(f"OFFICIAL SNAPSHOT {args.model_id}@{args.revision} files={len(files)} bytes={total}", flush=True)

    def download(item: dict) -> dict:
        name = item["rfilename"]
        path = target / name
        if target.resolve() not in path.resolve().parents:
            raise ValueError(f"Unsafe remote filename: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get("lfs", {}).get("sha256") if item.get("lfs") else None
        size = item.get("size")
        if path.is_file() and path.stat().st_size == size:
            digest = sha256(path)
            if expected is None or expected == digest:
                print(f"EXISTING VERIFIED {name} {size}", flush=True)
                return {"path": name, "bytes": size, "sha256": digest}
        partial = path.with_name(path.name + ".partial")
        file_url = f"{endpoint}/{args.model_id}/resolve/{args.revision}/{urllib.parse.quote(name)}"
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "Handwriting-Research-Snapshot/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                start = time.monotonic()
                with urllib.request.urlopen(urllib.request.Request(file_url, headers=headers), timeout=120) as response:
                    resumed = response.status == 206 and offset > 0
                    received = offset if resumed else 0
                    digest = hashlib.sha256()
                    if resumed:
                        with partial.open("rb") as previous:
                            for chunk in iter(lambda: previous.read(8 * 1024 * 1024), b""):
                                digest.update(chunk)
                    next_report = received + 256 * 1024 * 1024
                    with partial.open("ab" if resumed else "wb") as stream:
                        while chunk := response.read(8 * 1024 * 1024):
                            stream.write(chunk)
                            digest.update(chunk)
                            received += len(chunk)
                            if received >= next_report:
                                print(f"DOWNLOAD {name} {received}/{size} elapsed={time.monotonic()-start:.1f}s", flush=True)
                                next_report += 256 * 1024 * 1024
                actual = digest.hexdigest()
                if received != size:
                    raise RuntimeError(f"Size mismatch: {name} expected {size}, received {received}")
                if expected is not None and actual != expected:
                    raise RuntimeError(f"SHA256 mismatch: {name}")
                os.replace(partial, path)
                print(f"VERIFIED {name} bytes={received} sha256={actual}", flush=True)
                return {"path": name, "bytes": received, "sha256": actual}
            except Exception as error:
                print(f"DOWNLOAD FAILURE {name} attempt={attempt+1}: {error!r}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 10))
        raise AssertionError("Unreachable")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(download, files))
    manifest = {"model_id": args.model_id, "revision": args.revision, "format": "official_diffusers", "files": records}
    (target / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("OFFICIAL SNAPSHOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
