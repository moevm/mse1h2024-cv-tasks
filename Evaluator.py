import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import classes
from random import random
import torch.optim as optim
from torchvision import models
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from dataset import DatasetInterface
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

#from dataset.ResizeImages import resize
from metrics.RecallChecker import RecallChecker
from metrics.ROCAUCChecker import ROCAUCChecker
from metrics.F1ScoreChecker import F1ScoreChecker
from metrics.AccuracyChecker import AccuracyChecker
from metrics.PrecisionChecker import PrecisionChecker
from torch.utils.data import DataLoader, SubsetRandomSampler
import warnings
warnings.filterwarnings('ignore')
from Model import model

import PIL
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
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.predictions_len = predictions_len
        self.ground_truth_len = ground_truth_len
        self.df = pd.read_csv(filename)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(self):
        """
        Perform evaluation.

        Returns:
            Dictionary containing evaluation metrics.
        """


        #calculate metrics
        #ground_truth = labels, prediction = images
        full_metrics = []
        for images, labels in self.dataloader:
            images = images.to(self.device).float()
            pred_labels = self.model(images)
            model_ans = []
            for el in pred_labels:
                max_el = max(list(el))
                model_ans.append(list(el).index(max_el))

            #print(labels)
            #print(model_ans)
            metrics = self.calculate_metrics(model_ans, list(labels))
            full_metrics.append(metrics)

        #count column mean of all got metrics from dataloader
        #print(np.mean(full_metrics, axis = 0))

        # Create DataLoader for the dataset

        # # Convert predictions and ground truth to lists
        # ground_truth_list = list(ground_truth)
        # predictions_list = list(predictions)
        #
        # # Convert predictions to numpy array
        # predictions_array = np.array(predictions_list)
        #
        # # Reshape predictions_array to ensure it's two-dimensional
        # predictions_array = predictions_array.reshape(-1, 1)
        #
        # # Calculate metrics
        # metrics = self.calculate_metrics(predictions_array, ground_truth_list)
        #
        return np.mean(full_metrics, axis = 0)
    
    def calculate_metrics(self, predictions, ground_truth):
        """
        Calculate evaluation metrics.

        Args:
            predictions: List of predicted labels.
            ground_truth: List of ground truth labels.

        Returns:
            Dictionary containing evaluation metrics.
        """


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
        # roc_auc_checker = ROCAUCChecker("ROC-AUC")
        # fprs, tprs, roc_aucs = roc_auc_checker.calculate_metric(predictions, ground_truth)
        # avg_roc_auc = np.mean(roc_aucs)
        # metric_values["ROC-AUC"] = avg_roc_auc
        # metric_interpretations["ROC-AUC"] = f"{avg_roc_auc:.2%}"


        # Calculate averages
        # average_precision = np.mean(list(metric_values.values()))
        # average_accuracy = np.mean(list(metric_values.values()))
        # average_recall = np.mean(list(metric_values.values()))
        # average_f1_score = np.mean(list(metric_values.values()))
        #average_roc_auc = np.mean(list(metric_values.values()))

        # Return metrics and interpretations as tuples
        # metrics = {
        #     **metric_values,
        #     **metric_interpretations,
        #     "Average Precision": average_precision,
        #     "Average Accuracy": average_accuracy,
        #     "Average Recall": average_recall,
        #     "Average F1-score": average_f1_score,
        #     #"Average ROC-AUC": average_roc_auc,
        #     #"ROC Curve": (fprs, tprs, avg_roc_auc)
        # }

        return [precision,accuracy, recall, f1_score]

def resize(path="r'./datasets/train-scene/train'", size = 150):


    f = r'./dataset/datasets/train-scene/train'
    for file in os.listdir(f):
        f_img = f + "/" + file
        img = Image.open(f_img)
        img = img.resize((150, 150))
        img.save(f_img)

# Example usage
if __name__ == "__main__":



    # RESIZE ALL IMAGES
    resize()



    # Load pre-trained ResNet18 model
    # model = models.resnet18(weights='IMAGENET1K_V1')
    # model.fc = nn.Linear(model.fc.in_features, 6)

    # Instantiate ModelEvaluator
    eva = ModelEvaluator(model, DatasetInterface("./dataset/datasets/train-scene/train.csv",
                                                 "./dataset/datasets/train-scene/train/"),
                         64, 8517, 8517, "./dataset/datasets/train-scene/train.csv")


    # Evaluate the model
    metrics = eva.evaluate()

    # Print evaluation metrics
    interpretation = ["Average Precision","Average Accuracy","Average Recall","Average F1-score"]
    for index,metric  in enumerate(metrics.tolist()):
        print(f"{interpretation[index]}: {metric}")
    #for metric_name, metric_value in metrics.items():
    #    print(f"{metric_name}: {metric_value}")

    #   # Plot ROC curve
    # roc_curve_data = metrics["ROC Curve"]
    # plt.figure(figsize=(10, 8))
    #
    # # Plot ROC curve for each class
    # for i in range(len(roc_curve_data[0])):
    #     fpr_class = roc_curve_data[0][i]
    #     tpr_class = roc_curve_data[1][i]
    #     roc_auc_class = roc_curve_data[2]
    #     plt.plot(fpr_class, tpr_class, lw=2, color='red', label=f'ROC curve (class {i + 1}) (AUC = {roc_auc_class:.2f})')
    #
    # # Plot the diagonal line (random classifier)
    # plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    #
    # # Set labels and title
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    #
    # # Customize legend position
    # plt.legend(loc="lower right")
    #
    # # Show grid
    # plt.grid(True)
    #
    # # Set axis limits
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    #
    # # Show plot
    # plt.show()
