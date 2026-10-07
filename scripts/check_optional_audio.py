#!/usr/bin/env python3
"""Check the installed optional audio ABI without using audio in this experiment."""

import torchaudio
import torch

print(f"OPTIONAL_AUDIO_IMPORT_PASSED torch={torch.__version__} torchaudio={torchaudio.__version__}", flush=True)
