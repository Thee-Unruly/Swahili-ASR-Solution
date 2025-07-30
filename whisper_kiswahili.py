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