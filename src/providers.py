import json
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from captcha.image import ImageCaptcha

import src.config as config
from src.codec import Codec
from src.generator import SampleGenerator
from src.dataset import ImageDataset

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RUNS = ROOT / "runs"

# =================================================================================
# LATEST CONFIG SAMPLE GENERATOR
# =================================================================================

latest_run_file = max(RUNS.glob("*.json"), key = lambda file: int(file.stem))
with open(latest_run_file, "r") as file:
    latest_data = json.load(file)["config"]

captcha_generator = ImageCaptcha(
    width = latest_data["width"],
    height = latest_data["height"],
    font_sizes = latest_data["font_sizes"]
)
latest_config_generator = SampleGenerator(captcha_generator).generate

# =================================================================================
# DATASET GENERATOR
# =================================================================================

captcha_generator = ImageCaptcha(
    width = config.width,
    height = config.height,
    font_sizes = config.font_sizes
)
generator = SampleGenerator(captcha_generator).generate

transformer = transforms.ToTensor()

encoder = Codec().encode

decoder = Codec().decode

def create_dataset():
    return ImageDataset(generator, transformer, encoder)

# =================================================================================
# TRAINING PROVIDERS
# =================================================================================

def optimizer(model):
    return getattr(optim, config.nn_optimizer)(model.parameters(), lr=config.learning_rate)

def loss_function():
    return getattr(nn, config.nn_loss_function)()

# =================================================================================
# TIME FORMATER
# =================================================================================

def time_format(duration):
    hours = int(duration//3600)
    minutes = int((duration%3600)//60)
    seconds = int(duration%60)

    if hours == 0:
        if minutes == 0:
            return f"{seconds:02d}s"
        else:
            return f"{minutes:02d}m {seconds:02d}s"
    else:
        return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"