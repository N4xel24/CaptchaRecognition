from torch.utils.data import Dataset
import src.config as config

class ImageDataset(Dataset):
    def __init__(self, generator, transformer, encoder):
        self.transformer = transformer
        self.encoder = encoder
        self.image_file_set = []
        self.label_set = []

        for _ in range(config.buffer_size):
            image_file, label = generator()
            self.image_file_set.append(image_file)
            self.label_set.append(label)

    def __len__(self):
        return config.buffer_size
    
    def __getitem__(self, idx):
        image_file = self.image_file_set[idx]
        image = self.transformer(image_file)

        label = self.label_set[idx]
        encoded_label = self.encoder(label)

        return image, encoded_label