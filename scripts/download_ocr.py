#!/usr/bin/env python3
"""Download hash-pinned Chinese OCR models into this project only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


MODELS = {
    "ch_PP-OCRv5_det_mobile.onnx": ("PP-OCRv5/det", "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"),
    "ch_PP-OCRv5_rec_mobile.onnx": ("PP-OCRv5/rec", "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5"),
    "ch_PP-OCRv4_rec_server.onnx": ("PP-OCRv4/rec", "6a2676219be9907c7fc9cf61ebaa843bf2898777def567925b78886fcd90c07a"),
}


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "models/ocr"
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (folder, expected) in MODELS.items():
        path = target / name
        url = f"https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/{folder}/{name}"
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            print(f"VERIFIED EXISTING {name}", flush=True)
        else:
            partial = path.with_name(name + ".partial")
            for attempt in range(4):
                try:
                    print(f"DOWNLOAD {name} attempt={attempt+1}", flush=True)
                    digest = hashlib.sha256()
                    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                            digest.update(chunk)
                    if digest.hexdigest() != expected:
                        raise RuntimeError(f"OCR SHA256 mismatch: {name}")
                    os.replace(partial, path)
                    break
                except Exception as error:
                    print(f"DOWNLOAD FAILURE {name}: {error!r}", flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        manifest.append({"path": name, "url": url, "sha256": expected, "bytes": path.stat().st_size})
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("OCR MODELS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
