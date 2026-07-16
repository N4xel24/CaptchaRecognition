from torch.utils.data import Dataset
import src.config as config

class ImageDataset(Dataset):
    def __init__(self, generator, transformer, encoder):
        self.generator = generator
        self.transformer = transformer
        self.encoder = encoder

    def __len__(self):
        return config.epoch_size
    
    def __getitem__(self, idx):
        image, label = self.generator()

        image = self.transformer(image)
        label = self.encoder(label)

        return image, label