import json
from pathlib import Path

import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src.model import CaptchaCNN
import src.config as config
import src.providers as prov

start_time = time.time()

run_ID = datetime.now().strftime("%Y%m%d%H%M%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Run ID: {run_ID}")
print(f"Using device: {device}")

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RUNS = ROOT / "runs"

model = CaptchaCNN()
model.to(device)
optimizer = prov.optimizer(model)
criterion = prov.loss_function()
# Initialize training components

best_loss = float("inf")
best_epoch = 0

print("START TRAINING ...")
model.train()
for epoch in range(config.epochs):
    epoch_start = time.time()

    dataset = prov.create_dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=config.prefetch_factor
    )

    total_loss = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(images).view(-1, len(config.charset))
        loss = criterion(predictions, labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    average_loss = total_loss/len(dataloader)

    if average_loss < best_loss:
        best_loss = average_loss
        best_epoch = epoch+1
        torch.save(model.state_dict(), MODELS / f"{run_ID}.pth")
        
    epoch_end = time.time()
    epoch_exact_time = epoch_end - epoch_start
    epoch_time = prov.time_format(epoch_exact_time)

    if epoch_exact_time<60:
        print(f"[EPOCH: {epoch+1}/{config.epochs}] - Time = {epoch_exact_time:07.4f}s - Loss = {average_loss:.9f}")
    else:
        print(f"[EPOCH: {epoch+1}/{config.epochs}] - Time = {epoch_time} - Loss = {average_loss:.9f}")

print("MODEL LEARNED SUCCESSFULLY!")
print(f"[BEST EPOCH: {best_epoch}/{config.epochs}] BEST LOSS = {best_loss:.9f}")

end_time = time.time()
train_time = end_time - start_time
formatted_train_time = prov.time_format(train_time)

if train_time<60:
    print(f"TRAINING TIME = {train_time:07.4f}s")
else:
    print(f"TRAINING TIME = {formatted_train_time}")

result = {
    "best_epoch": best_epoch,
    "best_loss": best_loss,
    "train_time": formatted_train_time,
    "exact_time": train_time
}
run_info = {
    "config": config.config,
    "result": result
}

with open(RUNS / f"{run_ID}.json", "w") as file:
    json.dump(run_info, file, indent=4)
# Save training results