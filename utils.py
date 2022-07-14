import torch
import torchvision.transforms as transforms

transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

def collate_fn(samples):
    images = []
    labels = []
    for sample in samples:
        image, label = sample
        image = transform(image).unsqueeze(0)
        label = torch.tensor([label]).unsqueeze(0).float()
        images.append(image)
        labels.append(label)

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)

    return images, labels