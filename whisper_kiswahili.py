#!/usr/bin/env python3
"""
Kiswahili ASR Solution for Edge Deployment
Author: ASR Challenge Solution
"""

import torch
import torch.nn as nn
import torchaudio
import whisper
import pandas as pd
import numpy as np
from pathlib import Path
import librosa
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, Audio
import jiwer
from typing import Dict, List, Tuple, Optional
import logging
import time
import gc
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ASRConfig:
    """Configuration for ASR training and inference"""
    model_name: str = "openai/whisper-small"
    sample_rate: int = 16000
    max_audio_length: int = 30  # seconds
    batch_size: int = 16
    learning_rate: float = 1e-5
    num_epochs: int = 10
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 500
    max_memory_gb: int = 16
    target_memory_gb: int = 4  # Bonus target
    data_base_path: str = "C:/Users/ibrahim.fadhili/OneDrive - Agile Business Solutions/Desktop/ASR/kiswahili_asr/data/common_voice_swahili"

    