import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
class DatasetInterface(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        #print(img_path)
        return image, label

batch_size = 64

# example
if __name__ == '__main__':
    dataset = DatasetInterface("./datasets/train-scene/train.csv",
                            "./datasets/train-scene/train/")

    label = {
        0: "Buildings",
        1: "Forests",
        2: "Mountains",
        3: "Glacier",
        4: "Street",
        5: "Sea"
    }
    training_data, test_data = torch.utils.data.random_split(dataset, [17000, 34])
    train_loader = DataLoader(training_data, batch_size, True)
    test_loader = DataLoader(test_data, batch_size, True)

    train_features, train_label = next(iter(training_data))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_label}")
    img = train_features[0].squeeze()
    plt.imshow(img, cmap="gray")
    print(f"Label: {label[train_label]}")
    plt.show()
