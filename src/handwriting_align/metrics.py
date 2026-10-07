"""Exact text error counts and explicitly limited, non-learned style proxies."""

from __future__ import annotations

import math
import unicodedata

import cv2
import numpy as np
from PIL import Image


def on_white(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")
    return image.convert("RGB")


def normalize_text(text: str) -> str:
    """Apply NFKC and remove whitespace only; preserve punctuation and case."""
    return "".join(unicodedata.normalize("NFKC", text).split())


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, char in enumerate(first, 1):
        current = [i]
        for j, other in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def text_metrics(expected: str, recognized: str) -> dict[str, float | int]:
    expected, recognized = normalize_text(expected), normalize_text(recognized)
    if not expected:
        raise ValueError("Expected text must not be empty")
    errors = edit_distance(expected, recognized)
    return {"edits": errors, "characters": len(expected), "cer": errors / len(expected),
            "exact_match": int(expected == recognized), "content_score": max(0.0, 1.0 - errors / len(expected))}


def style_profile(image: Image.Image) -> np.ndarray | None:
    """Measure stroke/ink statistics, not writer identity or human preference."""
    gray = np.asarray(on_white(image).convert("L"))
    ink = (gray < 190).astype(np.uint8)
    ys, xs = np.nonzero(ink)
    if len(xs) < 16 or ink.mean() > 0.60:
        return None
    cropped = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    n, _labels, stats, _centers = cv2.connectedComponentsWithStats(cropped, 8)
    components = stats[1:][stats[1:, cv2.CC_STAT_AREA] >= 5]
    if not len(components):
        return None
    heights = components[:, cv2.CC_STAT_HEIGHT]
    widths = components[:, cv2.CC_STAT_WIDTH]
    scale = max(float(np.percentile(heights, 75)), 1.0)
    distance = cv2.distanceTransform(cropped, cv2.DIST_L2, 5)[cropped.astype(bool)]
    edges_x = np.abs(np.diff(cropped.astype(np.float32), axis=1)).sum()
    edges_y = np.abs(np.diff(cropped.astype(np.float32), axis=0)).sum()
    return np.array([np.median(distance) / scale, np.percentile(distance, 90) / scale,
                     np.median(widths / np.maximum(heights, 1)), float(cropped.mean()),
                     (float(edges_x) + 1.0) / (float(edges_y) + 1.0)], dtype=np.float64)


def style_similarity(image: Image.Image, reference: Image.Image) -> float:
    first, second = style_profile(image), style_profile(reference)
    if first is None or second is None:
        return 0.0
    difference = np.abs(np.log(np.maximum(first, 1e-5) / np.maximum(second, 1e-5)))
    return float(math.exp(-float(difference.mean())))


def combined_reward(expected: str, recognized: str, image: Image.Image, reference: Image.Image) -> dict:
    metrics = text_metrics(expected, recognized)
    style = style_similarity(image, reference)
    nonblank = style_profile(image) is not None
    content = metrics["content_score"] if nonblank else 0.0
    # Incorrect or blank text cannot earn a standalone style bonus.
    return {**metrics, "stroke_style_proxy": style, "nonblank": nonblank,
            "reward": content * (0.85 + 0.15 * style)}
