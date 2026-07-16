import json
from pathlib import Path
import torch
from PIL import Image

import src.providers as prov
from src.model import CaptchaCNN

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
MODELS = ROOT / "models"
INPUT = ROOT / "input"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CaptchaCNN()
model.to(device)

print("Choose an [ID] of the model you prefer:")

models = []
for path in MODELS.glob("*.pth"):
    models.append(path)

    with open(RUNS / f"{path.stem}.json", "r") as file:
        config = json.load(file)["config"]
        length = config["captcha_length"]
        width = config["width"]
        height = config["height"]
    print(f"[{len(models)}] {path.name} length={length}")

supported = [".png", ".jpg", ".jpeg"]
images = []
inputs = []

for format in supported:
    for path in (INPUT.glob(f"*{format}")):
        with Image.open(path) as file:
            images.append(path.name)
            image = file.convert("RGB")
            image = image.resize((width, height))
            image = prov.transformer(image)
            inputs.append(image)

model_id = int(input("Your model [ID]: ")) - 1
model.load_state_dict(torch.load(models[model_id], map_location=device))
model.eval()

with torch.no_grad():
    for i in range(len(inputs)):
        prediction = model(inputs[i].to(device).unsqueeze(0))
        prediction = prov.decoder(prediction)
        print(f"Model prediction of {images[i]}: {prediction}")