import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataset import DatasetInterface

batch_size = 64
if __name__ == '__main__':
    dataset = DatasetInterface("resources/archive/train-scene classification/train.csv",
                               "resources/archive/train-scene classification/train/")

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
