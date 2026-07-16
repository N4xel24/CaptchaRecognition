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

    epochs           = config["epochs"]
    epoch_size       = config["epoch_size"]
    batch_size       = config["batch_size"]
    num_workers      = config["num_workers"]
    prefetch_factor  = config["prefetch_factor"]

    learning_rate    = config["learning_rate"]
    nn_optimizer     = config["optimizer"]
    nn_loss_function = config["loss_function"]