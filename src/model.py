import torch.nn as nn
import src.config as config

width = config.width
height = config.height
for _ in range(3):
    width = (width-2)//2
    height = (height-2)//2

flatten_size = width * height * 128
output_size = config.captcha_length * len(config.charset)

class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()

        global flatten_size, output_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(flatten_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x