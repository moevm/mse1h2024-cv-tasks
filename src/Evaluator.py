import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import classes
from random import random
import torch.optim as optim
from torchvision import models
from sklearn.metrics import auc
from dataset import DatasetInterface
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from metrics.RecallChecker import RecallChecker
from metrics.ROCAUCChecker import ROCAUCChecker
from metrics.F1ScoreChecker import F1ScoreChecker
from metrics.AccuracyChecker import AccuracyChecker
from metrics.PrecisionChecker import PrecisionChecker
from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.metrics import roc_curve, auc

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
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.predictions_len = predictions_len
        self.ground_truth_len = ground_truth_len
        self.df = pd.read_csv(filename)

    def evaluate(self):
        """
        Perform evaluation.

        Returns:
            Dictionary containing evaluation metrics.
        """
        # Create DataLoader for the dataset
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        # Split the dataset into predictions and ground truth
        predictions, ground_truth = torch.utils.data.random_split(self.df["label"], [self.predictions_len, self.ground_truth_len])

        # Convert predictions and ground truth to lists
        ground_truth_list = list(ground_truth)
        predictions_list = list(predictions)

        # Convert predictions to numpy array
        predictions_array = np.array(predictions_list)

        # Reshape predictions_array to ensure it's two-dimensional
        predictions_array = predictions_array.reshape(-1, 1)

        # Calculate metrics
        metrics = self.calculate_metrics(predictions_array, ground_truth_list)

        return metrics

    def calculate_metrics(self, predictions, ground_truth):
        """
        Calculate evaluation metrics.

        Args:
            predictions: List of predicted labels.
            ground_truth: List of ground truth labels.

        Returns:
            Dictionary containing evaluation metrics.
        """
        # Initialize containers for metric values and interpretations
        metric_values = {}
        metric_interpretations = {}

        # Precision
        precision_checker = PrecisionChecker("Precision", average='macro')
        precision = precision_checker.calculate_metric(predictions, ground_truth)
        metric_values["Precision"] = precision
        metric_interpretations["Precision"] = f"{precision:.2%}"

        # Accuracy
        accuracy_checker = AccuracyChecker("Accuracy")
        accuracy = accuracy_checker.calculate_metric(predictions, ground_truth)
        metric_values["Accuracy"] = accuracy
        metric_interpretations["Accuracy"] = f"{accuracy:.2%}"

        # Recall
        recall_checker = RecallChecker("Recall", average='macro')
        recall = recall_checker.calculate_metric(predictions, ground_truth)
        metric_values["Recall"] = recall
        metric_interpretations["Recall"] = f"{recall:.2%}"

        # F1-score
        f1_score_checker = F1ScoreChecker("F1-score", average='macro')
        f1_score = f1_score_checker.calculate_metric(predictions, ground_truth)
        metric_values["F1-score"] = f1_score
        metric_interpretations["F1-score"] = f"{f1_score:.2%}"

        # ROC-AUC for each class
        # roc_auc_checker = ROCAUCChecker("ROC-AUC")
        # fprs, tprs, roc_aucs = roc_auc_checker.calculate_metric(predictions, ground_truth_one_hot)
        # avg_roc_auc = np.mean(roc_aucs)
        # metric_values["Average ROC-AUC"] = avg_roc_auc
        # metric_interpretations["Average ROC-AUC"] = f"{avg_roc_auc:.2%}"


        # Calculate averages
        average_precision = np.mean(list(metric_values.values()))
        average_accuracy = np.mean(list(metric_values.values()))
        average_recall = np.mean(list(metric_values.values()))
        average_f1_score = np.mean(list(metric_values.values()))
        #average_roc_auc = np.mean(list(metric_values.values()))

        # Return metrics and interpretations as tuples
        metrics = {
            **metric_values,
            **metric_interpretations,
            "Average Precision": average_precision,
            "Average Accuracy": average_accuracy,
            "Average Recall": average_recall,
            "Average F1-score": average_f1_score,
            #"Average ROC-AUC": average_roc_auc,
            #"ROC Curve": (fpr, tpr, roc_auc)
        }

        return metrics


# Example usage
if __name__ == "__main__":
    # Load pre-trained ResNet18 model
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 6)

    # Instantiate ModelEvaluator
    # Instantiate ModelEvaluator
    # Instantiate ModelEvaluator
    eva = ModelEvaluator(model, DatasetInterface("./src/dataset/dataset/datasets/train-scene/train.csv",
                                             "./src/dataset/datasets/train-scene/train/"),
                     64, 8517, 8517, "./src/dataset/dataset/datasets/train-scene/train.csv")


    # Evaluate the model
    metrics = eva.evaluate()

    # Print evaluation metrics
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    # # Plot ROC curve
    # fpr, tpr, roc_auc = metrics["ROC Curve"]
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic Curve')
    # plt.legend(loc="lower right")
    # plt.show()
