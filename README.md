# Captcha Recognition

A beginner-friendly CAPTCHA recognition project built with **PyTorch**.

The purpose of this project is to learn the complete deep learning workflow, including synthetic dataset generation, CNN training, model saving, and inference on new CAPTCHA images.

---

## Features

- Generate synthetic CAPTCHA images
- Train a CNN model for CAPTCHA recognition
- Generate datasets on the fly during training
- Save the best model automatically
- Save training configurations and results
- Predict randomly generated CAPTCHA images
- Solve user-provided CAPTCHA images

---

## Project Structure

```
captcha_recognition/
├── config.json
├── data/
├── input/
├── models/
├── runs/
├── eval/
│   └── accuracy.py
├── src/
│   ├── codec.py
│   ├── config.py
│   ├── dataset.py
│   ├── generator.py
│   ├── model.py
│   ├── predict.py
│   ├── providers.py
│   ├── solve.py
│   └── train.py
└── README.md
```

---

## Installation

Clone the repository.

```bash
git clone <repository-url>
cd captcha_recognition
```

Install the required packages.

```bash
pip install -r requirements.txt
```

---

## Configuration

All training settings are stored in `config.json`.

Example:

```json
{
    "width":            160,
    "height":           64,
    "font_sizes":       [30, 35, 40, 45, 50, 55, 60, 65, 70],
    "charset":          "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "captcha_length":   1,

    "epochs":           30,
    "epoch_size":       1000,
    "batch_size":       32,
    "num_workers":      4,
    "prefetch_factor":  4,

    "learning_rate":    0.001,
    "optimizer":        "Adam",
    "loss_function":    "CrossEntropyLoss"
}
```

You can change these values before training.

---

## Training

Run

```bash
python -m src.train
```

During training the project will

- generate CAPTCHA images
- train the CNN model
- save the best model in `models/`
- save the training information in `runs/`

Example output

```text
[EPOCH: 1/30] - Time = 01h 16m 53s - Loss = 0.473471284
[EPOCH: 2/30] - Time = 01h 16m 53s - Loss = 0.085300421
[EPOCH: 3/30] - Time = 01h 16m 59s - Loss = 0.054018025
```

---

## Predict a Random CAPTCHA

Run

```bash
python -m src.predict
```

The program will

- generate a new CAPTCHA image
- let the model predict it
- print both the prediction and the correct label

Example

```text
Model prediction: UQK
Target (Label):   UQK
```

---

## Solve Your Own CAPTCHA Images

Put your images into the `input/` folder.

Supported image formats:

- `.png`
- `.jpg`
- `.jpeg`

Run

```bash
python -m src.solve
```

Choose one of the trained models.

Example

```text
Choose an [ID] of the model you prefer:

[1] 20260713013036.pth
[2] 20260714022324.pth

Your model [ID]: 2
```

The program will resize each image, run inference, and print the prediction.

Example

```text
Model prediction of OWN.png: DWN
Model prediction of UQK.png: UQK
Model prediction of QTC.png: QTC
```

---

## Model

The CNN architecture consists of

```
Input
    ↓
Conv2D
    ↓
BatchNorm
    ↓
ReLU
    ↓
MaxPool

    ↓

Conv2D
    ↓
BatchNorm
    ↓
ReLU
    ↓
MaxPool

    ↓

Conv2D
    ↓
BatchNorm
    ↓
ReLU
    ↓
MaxPool

    ↓

Flatten
    ↓
Linear
    ↓
BatchNorm
    ↓
ReLU
    ↓
Linear
```

---

## Dataset

This project does **not** store a fixed dataset for training.

Instead, CAPTCHA images are generated **on the fly**.

This allows the model to continuously see new CAPTCHA samples during training.

---

## Training Logs

Each training run creates

- a model checkpoint in `models/`
- a JSON log file in `runs/`

The log file stores

- training configuration
- best epoch
- best loss
- training time

---

## Future Improvements

Some possible future improvements are

- Data augmentation
- More CAPTCHA styles
- Better preprocessing
- Transfer learning
- Web interface

---

## Acknowledgements

This project uses

- PyTorch
- torchvision
- Pillow
- captcha

---

## Notes

This is one of my first computer vision projects.

The main goal of this project is to better understand the complete workflow of building and training a deep learning model, from data generation to inference.