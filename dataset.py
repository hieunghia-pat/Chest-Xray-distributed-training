from torch.utils.data import Dataset

import os
from PIL import Image, ImageOps
import json

class ChestXRayDataset(Dataset):
    def __init__(self, image_path, annotation_path):
        self.image_path = image_path

        self.data = json.load(open(annotation_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        annotation = self.data[idx]
        filename = annotation["filename"]
        filepath = annotation["filepath"]
        label = annotation["label"]

        image = Image.open(os.path.join(self.image_path, filepath, filename)).convert("RGB")

        return image, label