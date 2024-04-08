import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from random import random
import torch.optim as optim
from torchvision import models
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from dataset import DatasetInterface
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from metrics.RecallChecker import RecallChecker
from metrics.ROCAUCChecker import ROCAUCChecker
from metrics.F1ScoreChecker import F1ScoreChecker
from metrics.AccuracyChecker import AccuracyChecker
from metrics.PrecisionChecker import PrecisionChecker
from torch.utils.data import DataLoader, SubsetRandomSampler
import warnings
warnings.filterwarnings('ignore')
from models.Model import model

import os
from PIL import Image

# Define a class for evaluating a model
class ModelEvaluator:
    def __init__(self, model, dataset, batch_size, predictions_len, ground_truth_len, filename):
        """
        Initialize the ModelEvaluator.

        Args:
            model: The model to be evaluated.
            dataset: The dataset to be used for evaluation.
            batch_size: Batch size for DataLoader.
            predictions_len: Length of predictions for splitting dataset.
            ground_truth_len: Length of ground truth for splitting dataset.
            filename: CSV filename.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Define device
        self.model = model.to(self.device)  # Move the model to the specified device
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.predictions_len = predictions_len
        self.ground_truth_len = ground_truth_len
        self.df = pd.read_csv(filename)

    def evaluate(self):
        """
        Perform evaluation.
    
        Returns:
            List containing evaluation metrics.
        """
        # Set the model to evaluation mode
        self.model.eval()
    
        full_metrics = []
        predictions = []
        ground_truth = []
        for images, labels in self.dataloader:
            images = images.to(self.device).float()  # Move images to the same device as the model and ensure float data type
            labels = labels.to(self.device)  # Move labels to the same device as the model
            with torch.no_grad():  # Disable gradient tracking during inference
                pred_labels = self.model(images)
            
            model_ans = []
            for el in pred_labels:
                max_el = max(list(el))
                model_ans.append(list(el).index(max_el))
            
            # Move model predictions and labels to CPU before calculating metrics
            model_ans_cpu = torch.tensor(model_ans, device='cpu')
            labels_cpu = labels.cpu()
    
            metrics, fpr, tpr= self.calculate_metrics(model_ans_cpu, labels_cpu)
            full_metrics.append(metrics)
            predictions.extend(model_ans)
            ground_truth.extend(labels_cpu)
        
        return np.mean(full_metrics, axis=0), fpr, tpr

    def calculate_metrics(self, predictions, ground_truth):
        """
        Calculate evaluation metrics.

        Args:
            predictions: List of predicted labels.
            ground_truth: List of ground truth labels.

        Returns:
            List containing evaluation metrics.
        """
        # Convert predictions to numpy array
        predictions = np.array(predictions)

        # Move ground truth tensor to CPU and convert to numpy array
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu().detach().numpy()
        else:
            ground_truth = np.array(ground_truth)

        # Precision
        precision_checker = PrecisionChecker("Precision", average='macro')
        precision = precision_checker.calculate_metric(predictions, ground_truth)

        # Accuracy
        accuracy_checker = AccuracyChecker("Accuracy")
        accuracy = accuracy_checker.calculate_metric(predictions, ground_truth)

        # Recall
        recall_checker = RecallChecker("Recall", average='macro')
        recall = recall_checker.calculate_metric(predictions, ground_truth)

        # F1-score
        f1_score_checker = F1ScoreChecker("F1-score", average='macro')
        f1_score = f1_score_checker.calculate_metric(predictions, ground_truth)

        # ROC-AUC for each class
        roc_auc_checker = ROCAUCChecker("ROC-AUC")
        fpr, tpr, roc_auc = roc_auc_checker.calculate_metric(predictions, ground_truth)
        
        # Ensure all metrics have fixed lengths
        fixed_metrics = [precision, accuracy, recall, f1_score, roc_auc]

        # Ensure all metrics have fixed lengths
        fixed_metrics = [precision, accuracy, recall, f1_score, roc_auc]

        return fixed_metrics, fpr, tpr

def resize(path="./dataset/datasets/train-scene/train", size=150):
    """
    Resize images in a directory.

    Args:
        path: Path to the directory containing images.
        size: Size to resize the images.
    """
    for file in os.listdir(path):
        f_img = os.path.join(path, file)
        img = Image.open(f_img)
        img = img.resize((size, size))
        img.save(f_img)


if __name__ == "__main__":
    # Instantiate ModelEvaluator
    eva = ModelEvaluator(model, DatasetInterface("./dataset/datasets/train-scene/train.csv",
                                                 "./dataset/datasets/train-scene/train/"),
                         64, 8517, 8517, "./dataset/datasets/train-scene/train.csv")

    # Evaluate the model
    metrics, fpr, tpr = eva.evaluate()

    # Print evaluation metrics
    interpretation = ["Average Precision", "Average Accuracy", "Average Recall", "Average F1-score", "Average ROC-AUC"]
    for index, metric in enumerate(metrics.tolist()):
        print(f"{interpretation[index]}: {metric}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')  # Use a contrasting color and thicker line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)  # Add gridlines
    plt.show()
