import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader


class DatasetInterface(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        Custom dataset class for image classification.
        
        Args:
        annotations_file (str): Path to the CSV file containing image paths and labels.
        img_dir (str): Directory containing the images.
        transform (callable, optional): Optional transform to be applied to the image.
        target_transform (callable, optional): Optional transform to be applied to the label.
        """
        self.img_labels = pd.read_csv(annotations_file, index_col=0) # Load image labels from CSV file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.img_labels)

    def __getitem__(self, index):
        """Returns a tuple (image, label) for the given index."""
        image_name = str(self.img_labels.iloc[index, 0])
        image_name = '0' * (8 - len(image_name)) + image_name 
        img_path = os.path.join(self.img_dir,image_name) # Get image path
        image = read_image(img_path) # Read image using torchvision's read_image function

        label = self.img_labels.iloc[index, 1] # Get label for the image

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

batch_size = 64

# example
if __name__ == '__main__':
    # Create dataset instance
    dataset = DatasetInterface("./datasets/train-scene/train.csv",
                            "./datasets/train-scene/train/")

    # Define labels for the dataset
    label = {
        0: "Buildings",
        1: "Forests",
        2: "Mountains",
        3: "Glacier",
        4: "Street",
        5: "Sea"
    }

    # Split dataset into training and test sets
    training_data, test_data = torch.utils.data.random_split(dataset, [17000, 34])

    # Create data loaders for training and test sets
    train_loader = DataLoader(training_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)

    # Retrieve a batch of features and labels from the training set
    train_features, train_label = next(iter(training_data))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_label}")

    # Display an image from the batch with its label
    img = train_features[0].squeeze()
    plt.imshow(img, cmap="gray")
    print(f"Label: {label[train_label]}")
    plt.show()
