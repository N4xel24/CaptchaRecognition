import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "config.json", "r") as file:
    config = json.load(file)

    width            = config["width"]
    height           = config["height"]
    font_sizes       = config["font_sizes"]
    charset          = config["charset"]
    captcha_length   = config["captcha_length"]
    epoch_size       = config["epoch_size"]
    buffer_size      = config["buffer_size"]
    batch_size       = config["batch_size"]
    learning_rate    = config["learning_rate"]
    epochs           = config["epochs"]
    nn_optimizer     = config["optimizer"]
    nn_loss_function = config["loss_function"]

if epoch_size % buffer_size != 0:
    raise ValueError("epoch_size must be divisible by buffer_size")