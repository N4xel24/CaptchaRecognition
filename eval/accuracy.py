import torch
from pathlib import Path

import src.providers as prov
from src.model import CaptchaCNN

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
MODELS = ROOT / "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CaptchaCNN()
model.to(device)
latest_pth_file = max(MODELS.glob("*.pth"), key = lambda file: int(file.stem))
model.load_state_dict(torch.load(latest_pth_file, map_location=device))
model.eval()

correct = 0
for _ in range(10000):
    image, label = prov.latest_config_generator()

    image = image.convert("RGB")
    image = prov.transformer(image).unsqueeze(0)
    # Convert (C, H, W) to (1, C, H, W).
    # The model expects a batch dimension of images.

    image = image.to(device)

    with torch.no_grad():
        prediction = model(image)
        prediction = prov.decoder(prediction)

    if prediction==label:
        correct+=1

accuracy = correct/10000
print(f"Model Accuracy: {accuracy:.2%}")