#!/usr/bin/env python3
"""Copy a reproducible native-line subset into a fully independent project."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import unicodedata

import numpy as np
from PIL import Image, ImageOps


def canonical(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def prompt(text: str) -> str:
    return ("请参考输入图像中书写者的笔迹、笔画粗细和字形，在纯白背景上用相同手写风格生成一行文字。"
            "只写下面引号内的内容，逐字准确，不要解释，不要添加标题，不要抄写参考图中的内容。"
            "字间距自然、文字清晰，保留所有标点和数字。需要书写的文字是：\"" + text + "\"。")


def line_image(source: Path, target: Path, size: tuple[int, int]) -> None:
    image = Image.open(source).convert("L")
    ink = np.asarray(image) < 210
    ys, xs = np.nonzero(ink)
    if not len(xs):
        raise ValueError(f"Blank native line: {source}")
    image = image.crop((max(0, int(xs.min()) - 4), max(0, int(ys.min()) - 4),
                        min(image.width, int(xs.max()) + 5), min(image.height, int(ys.max()) + 5)))
    image = ImageOps.contain(image, (size[0] - 48, size[1] - 80), Image.Resampling.LANCZOS)
    page = Image.new("RGB", size, "white")
    page.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    target.parent.mkdir(parents=True, exist_ok=True)
    page.save(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--dev-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--max-characters", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / "data/native_subset"
    if (data / "manifest.json").exists():
        raise FileExistsError("A prepared dataset already exists; do not silently redefine an experiment")
    source = args.source_project.resolve()
    original = json.loads((source / "data/prepared/native_lines_v1/manifest.json").read_text(encoding="utf-8"))
    candidates = [row for row in original["samples"] if row["kind"] == "native_line" and 3 <= len(canonical(row["text"])) <= args.max_characters]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    counts = Counter(row["split"] for row in candidates)
    print(f"Native eligible lines (3..{args.max_characters} chars): {dict(counts)}", flush=True)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        if row["split"] == "train":
            buckets[row["style_id"]].append(row)
    styles = sorted(buckets, key=lambda name: (-len(buckets[name]), name))[:128]
    pool = {name: list(buckets[name]) for name in styles}
    train = []
    texts: set[str] = set()
    while len(train) < args.train_count and any(pool.values()):
        for style in styles:
            while pool[style]:
                row = pool[style].pop()
                normalized = canonical(row["text"])
                if normalized not in texts:
                    train.append(row)
                    texts.add(normalized)
                    break
            if len(train) == args.train_count:
                break
    if len(train) != args.train_count:
        raise ValueError(f"Insufficient distinct native short lines among 128 writers: {len(train)} < {args.train_count}")
    train_writers = {row["style_id"] for row in train}
    train_pages = {row["source_page"] for row in train}
    reserved_pages: set[str] = set()

    def choose(count: int) -> list[dict]:
        selected = []
        for wanted, split in ((count // 2, "validation_seen"), (count - count // 2, "validation_unseen")):
            part = []
            for row in candidates:
                if row["split"] != split or (split == "validation_seen" and row["style_id"] not in train_writers):
                    continue
                normalized = canonical(row["text"])
                if normalized in texts or row["source_page"] in train_pages or row["source_page"] in reserved_pages:
                    continue
                part.append(row)
                texts.add(normalized)
                if len(part) == wanted:
                    break
            if len(part) != wanted:
                raise ValueError(f"Insufficient page-separated {split} cases: {len(part)} < {wanted}")
            selected.extend(part)
        reserved_pages.update(row["source_page"] for row in selected)
        return selected

    development = choose(args.dev_count)
    test = choose(args.test_count)
    selections = {"train": train, "development": development, "test": test}
    all_rows = []
    hashes = []
    copied: set[Path] = set()
    for split, rows in selections.items():
        output = []
        for row in rows:
            target_rel = f"targets/{row['stem']}.png"
            reference_rel = f"references/{row['style_id']}.png"
            line_image(source / row["image"], data / target_rel, (768, 256))
            reference = data / reference_rel
            if reference not in copied:
                reference.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / row["reference_image"], reference)
                copied.add(reference)
            provenance = {}
            for key in ("source_page", "reference_page"):
                page_rel = "provenance/pages/" + str(Path(row[key]).relative_to("data/raw/CASIA_HWDB_LINES/pages"))
                page = data / page_rel
                if page not in copied:
                    page.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / row[key], page)
                    copied.add(page)
                provenance[key] = page_rel
            record = {"id": row["stem"], "text": row["text"], "prompt": prompt(row["text"]), "style_id": row["style_id"],
                      "original_split": row["split"], "split": split, "target": target_rel, "reference": reference_rel,
                      "line_index": row["line_index"], **provenance}
            if provenance["source_page"] == provenance["reference_page"]:
                raise AssertionError("Reference and target are on the same source page")
            output.append(record)
            all_rows.append(record)
        (data / f"{split}_cases.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for mode in ("sft", "grpo"):
        folder = data / mode
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "train.jsonl").open("w", encoding="utf-8") as stream:
            for row in all_rows:
                if row["split"] != "train":
                    continue
                metadata = {"id": row["id"], "target_text": row["text"], "style_id": row["style_id"]}
                if mode == "sft":
                    record = {"schema_version": 2, "input": {"prompt": row["prompt"], "media": [{"type": "image", "path": "../" + row["reference"]}]},
                              "supervision": {"type": "demonstration", "target": {"media": [{"type": "image", "path": "../" + row["target"]}]}}, "metadata": metadata}
                else:
                    record = {"prompt": row["prompt"], "images": ["../" + row["reference"]], **metadata}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    for row in all_rows[:4]:
        Image.open(data / row["target"]).verify()
    for path in sorted(set(copied) | {data / row["target"] for row in all_rows}):
        hashes.append({"path": str(path.relative_to(data)), "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pages = {split: {r["source_page"] for r in all_rows if r["split"] == split} for split in selections}
    if any(pages[a] & pages[b] for a, b in (("train", "development"), ("train", "test"), ("development", "test"))):
        raise AssertionError("Target page leakage across dataset partitions")
    manifest = {"format": "native-line-alignment-v1", "seed": args.seed, "target_size": [768, 256],
                "counts": {s: len(v) for s, v in selections.items()}, "train_writers": len(train_writers),
                "eligible_source_counts": dict(counts), "distinct_train_characters": len(set("".join(r["text"] for r in train))),
                "partition_checks": {"normalized_text_overlap": 0, "target_page_overlap": 0, "reference_on_target_page": 0},
                "files": hashes, "samples": all_rows}
    (data / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for mode in ("sft", "grpo"):
        smoke = data / f"{mode}_smoke"
        smoke.mkdir(exist_ok=True)
        training_rows = [row for row in all_rows if row["split"] == "train"]
        smoke_ids = {row["id"] for row in sorted(training_rows, key=lambda row: (len(row["text"]), row["id"]))[:8]}
        all_lines = (data / mode / "train.jsonl").read_text(encoding="utf-8").splitlines()
        lines = [line for row, line in zip(training_rows, all_lines) if row["id"] in smoke_ids]
        (smoke / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PREPARED counts={manifest['counts']} writers={len(train_writers)} characters={manifest['distinct_train_characters']} provenance_files={len(copied)}", flush=True)


if __name__ == "__main__":
    main()
