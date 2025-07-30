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
    
    def _speed_perturbation(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """Apply speed perturbation"""
        if factor == 1.0:
            return waveform
        
        # Use librosa for speed change
        waveform_np = waveform.numpy()
        waveform_stretched = librosa.effects.time_stretch(waveform_np, rate=factor)
        return torch.from_numpy(waveform_stretched)
    
class SwahiliDataset:
    """Dataset handler for Kiswahili speech data"""
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self.processor = AudioProcessor(config)
        
    def prepare_common_voice_data(self, data_path: str) -> Dataset:
        """Prepare Mozilla Common Voice Swahili dataset using preprocessed data"""
        logger.info("Loading preprocessed Common Voice Swahili dataset...")
        
        # Load cleaned validated data
        validated_df = pd.read_csv(f"{data_path}/tsv_cleaned/validated_cleaned.tsv", sep='\t')
        
        # Map audio paths to augmented WAV files
        validated_df["audio"] = validated_df["path"].apply(
            lambda x: f"{data_path}/wav_augmented/{Path(x).stem}.wav"
        )
        
        # Prepare dataset
        dataset = Dataset.from_pandas(validated_df[["audio", "sentence"]])
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.config.sample_rate))
        
        # Optional: Split into train, val, test if not already separated
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, train_size + val_size + test_size))
        
        return train_dataset, val_dataset, test_dataset
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess Swahili text"""
        text = text.lower().strip()
        text = text.replace("'", "'")  # Normalize apostrophes
        text = ' '.join(text.split())
        return text
    
class OptimizedWhisperModel(nn.Module):
    """Optimized Whisper model for edge deployment"""
    
    def __init__(self, config: ASRConfig):
        super().__init__()
        self.config = config
        
        # Load base Whisper model
        self.whisper_model = whisper.load_model("small")
        
        # Enable gradient checkpointing for memory efficiency
        self.whisper_model.encoder.use_cache = False
        
    def forward(self, mel_spectrogram):
        """Forward pass with memory optimization"""
        return self.whisper_model.decode(mel_spectrogram)
    
    def transcribe_streaming(self, audio_chunk: torch.Tensor) -> str:
        """Streaming transcription for real-time processing"""
        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(audio_chunk)
            result = self.whisper_model.decode(mel)
            return result.text
        
class ModelOptimizer:
    """Model optimization for edge deployment"""
    
    def __init__(self, model: nn.Module, config: ASRConfig):
        self.model = model
        self.config = config
    
    def quantize_model(self) -> nn.Module:
        """Apply quantization for memory efficiency"""
        logger.info("Applying dynamic quantization...")
        
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv1d},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def prune_model(self, sparsity: float = 0.3) -> nn.Module:
        """Apply magnitude-based pruning"""
        logger.info(f"Applying pruning with {sparsity} sparsity...")
        
        import torch.nn.utils.prune as prune
        
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')
        
        return self.model
    
    def optimize_for_inference(self) -> nn.Module:
        """Comprehensive optimization for inference"""
        model = self.quantize_model()
        model = self.prune_model()
        model.eval()
        return model
    
class SwahiliASRTrainer:
    """Main trainer class for Swahili ASR"""
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self.dataset_handler = SwahiliDataset(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = OptimizedWhisperModel(config).to(self.device)
        
    def train(self):
        """Train the ASR model"""
        logger.info("Starting training...")
        
        # Prepare dataset using the preprocessed data path
        train_dataset, val_dataset, test_dataset = self.dataset_handler.prepare_common_voice_data(self.config.data_base_path)
        
        # Fine-tune Whisper model
        self._fine_tune_whisper(train_dataset, val_dataset)
        
        # Optimize model
        optimizer = ModelOptimizer(self.model, self.config)
        self.model = optimizer.optimize_for_inference()
        
        logger.info("Training completed!")

    def _fine_tune_whisper(self, train_dataset, val_dataset):
        """Fine-tune Whisper on Swahili data (simplified)"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Simplified training loop (needs proper implementation)
            for i in range(0, len(train_dataset), self.config.batch_size):
                batch = train_dataset[i:i + self.config.batch_size]
                audio_batch = torch.stack([self.dataset_handler.processor.load_audio(item["audio"]) for item in batch])
                text_batch = [self.dataset_handler.preprocess_text(item["sentence"]) for item in batch]
                
                # Placeholder for forward pass and loss (requires Whisper-specific handling)
                # This needs a proper data collator and loss function (e.g., CTC)
                optimizer.zero_grad()
                # ... (implement forward pass and loss here)
                # loss.backward()
                # optimizer.step()
            
            # Memory cleanup
            if (epoch + 1) % 2 == 0:
                torch.cuda.empty_cache()
                gc.collect()