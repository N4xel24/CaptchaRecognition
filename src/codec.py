import torch
import src.config as config

char2idx = {char:idx for idx, char in enumerate(config.charset)}
idx2char = {idx:char for idx, char in enumerate(config.charset)}

class Codec:
    def encode(self, text):
        """
        Encode labels into a tensor of indices for CrossEntropyLoss.
        """
        code = [char2idx[i]for i in text]
        return torch.tensor(code, dtype=torch.long)
    
    def decode(self, tensor):
        """
        Decode model predictions into readable text.
        """
        viewed_tensor = tensor.view(-1, len(config.charset))
        code = viewed_tensor.argmax(dim=1).tolist()
        decoded = [idx2char[idx] for idx in code]
        return "".join(decoded)