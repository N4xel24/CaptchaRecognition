from captcha.image import ImageCaptcha
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "input"

captcha = ImageCaptcha(
    width = 160,
    height = 64,
    font_sizes = [30, 35, 40, 45, 50, 55, 60, 65, 70]
)
for i in range(10):
    label = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    captcha.write(label, INPUT / f"{label}.png")