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

class AudioProcessor:
    """Handles audio preprocessing and augmentation"""
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:
            # Load audio using torchaudio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Normalize
            waveform = waveform / torch.max(torch.abs(waveform))
            
            return waveform.squeeze()
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return torch.zeros(self.sample_rate)  # Return silence on error
        
    def apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation techniques"""
        # Speed perturbation
        if np.random.random() < 0.3:
            speed_factor = np.random.uniform(0.9, 1.1)
            waveform = self._speed_perturbation(waveform, speed_factor)
        
        # Add noise
        if np.random.random() < 0.2:
            noise_factor = np.random.uniform(0.01, 0.05)
            noise = torch.randn_like(waveform) * noise_factor
            waveform = waveform + noise
        
        return waveform
    
    