"""CPU-only, project-local OCR; a second recognizer is reserved for evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from rapidocr import ModelType, OCRVersion, RapidOCR

from .metrics import on_white


class ChineseOCR:
    def __init__(self, independent_evaluator: bool = False):
        folder = Path(__file__).resolve().parents[2] / "models/ocr"
        detector = folder / "ch_PP-OCRv5_det_mobile.onnx"
        recognizer = folder / ("ch_PP-OCRv4_rec_server.onnx" if independent_evaluator else "ch_PP-OCRv5_rec_mobile.onnx")
        for path in (detector, recognizer):
            if not path.is_file():
                raise FileNotFoundError(f"Offline OCR model missing: {path}")
        self.engine = RapidOCR(params={
            "Global.model_root_dir": str(folder), "Global.use_cls": False,
            "Global.log_level": "error", "Global.text_score": 0.3,
            "Det.model_path": str(detector), "Det.ocr_version": OCRVersion("PP-OCRv5"), "Det.model_type": ModelType("mobile"),
            "Rec.model_path": str(recognizer), "Rec.ocr_version": OCRVersion("PP-OCRv4" if independent_evaluator else "PP-OCRv5"),
            "Rec.model_type": ModelType("server" if independent_evaluator else "mobile"),
            "EngineConfig.onnxruntime.use_cuda": False,
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        })

    def recognize(self, image: Image.Image) -> str:
        result = self.engine(np.asarray(on_white(image)))
        if result.txts is None:
            return ""
        if result.boxes is None:
            return "".join(result.txts)
        order = sorted(range(len(result.txts)), key=lambda i: (int(np.asarray(result.boxes[i])[:, 1].mean() // 20),
                                                              float(np.asarray(result.boxes[i])[:, 0].mean())))
        return "".join(result.txts[i] for i in order)
