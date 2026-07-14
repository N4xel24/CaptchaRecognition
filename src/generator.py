import src.config as config
import random

class SampleGenerator:
    """
    Generate a single CAPTCHA sample based on the project configuration from config.json.
    Each CAPTCHA sample includes one image and its corresponding label.
    """
    def __init__(self, captcha_generator):
        self.captcha_generator = captcha_generator

    def generate(self):
        label = "".join(random.choices(config.charset, k=config.captcha_length))
        image = self.captcha_generator.generate_image(label)

        return image, label