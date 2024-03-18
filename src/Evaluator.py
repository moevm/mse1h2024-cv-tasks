from random import random

import numpy as np
import torch
from sklearn.metrics import auc
from torch import classes
from torch.utils.data import DataLoader

from dataset import DatasetInterface

from metrics.AccuracyChecker import AccuracyChecker
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from torchvision import models


# from TestMetrics import calculate_metrics

class ModelEvaluator:
    def __init__(self, model, dataset, batch_size, predictions_len, ground_truth_len, filename):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.predictions_len = predictions_len
        self.ground_truth_len = ground_truth_len
        self.df = pd.read_csv(filename)

    def evaluate(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        predictions, ground_truth = torch.utils.data.random_split(self.df['label'], [self.predictions_len, self.ground_truth_len])
        #
        #
        # predictions_loader = DataLoader(predictions, self.batch_size, True)
        # ground_truth_loader = DataLoader(ground_truth, self.batch_size, True)
        new_data = self.df['label'].tolist()
        random.shuffle(new_data)
        train  = new_data[17000:]

        print(train)
        metrics = self.calculate_metrics(self.df['label'].tolist(),self.df['label'].tolist())

        return metrics

    def calculate_metrics(self, predictions, ground_truth):
        # Initialize containers for metric values and interpretations
        metric_values = {}
        metric_interpretations = {}

        # Precision
        # precision_checker = PrecisionChecker("Precision")
        # precision = precision_checker.calculate_metric(predictions, ground_truth)
        # metric_values["Precision"] = precision
        # metric_interpretations["Precision"] = f"{precision:.2%}"

        # Accuracy
        accuracy_checker = AccuracyChecker("Accuracy")
        accuracy = accuracy_checker.calculate_metric(predictions, ground_truth)
        metric_values["Accuracy"] = accuracy
        metric_interpretations["Accuracy"] = f"{accuracy:.2%}"

        # Recall
        # recall_checker = RecallChecker("Recall")
        # recall = recall_checker.calculate_metric(predictions, ground_truth)
        # metric_values["Recall"] = recall
        # metric_interpretations["Recall"] = f"{recall:.2%}"

        # F1-score
        # f1_score_checker = F1ScoreChecker("F1-score")
        # f1_score = f1_score_checker.calculate_metric(predictions, ground_truth)
        # metric_values["F1-score"] = f1_score
        # metric_interpretations["F1-score"] = f"{f1_score:.2%}"

        # ROC-AUC
        # roc_auc_checker = ROCAUCChecker("ROC-AUC")
        # fpr, tpr, thresholds = roc_auc_checker.calculate_metric(predictions, ground_truth)
        # roc_auc = auc(fpr, tpr)
        # metric_values["ROC-AUC"] = roc_auc
        # metric_interpretations["ROC-AUC"] = f"{roc_auc:.2%}"

        # Calculate averages
        # average_precision = np.mean(list(metric_values.values()))
        average_accuracy = np.mean(list(metric_values.values()))
        # average_recall = np.mean(list(metric_values.values()))
        # average_f1_score = np.mean(list(metric_values.values()))
        # average_roc_auc = np.mean(list(metric_values.values()))

        # Return metrics and interpretations as tuples
        metrics = {
            **metric_values,
            **metric_interpretations,
           # "Average Precision": average_precision,
            "Average Accuracy": average_accuracy,
            #"Average Recall": average_recall,
         #   "Average F1-score": average_f1_score,
         #   "Average ROC-AUC": average_roc_auc,
          #  "ROC Curve": (fpr, tpr, roc_auc)
        }

        return metrics



# Example usage
if __name__ == "__main__":
    # data = ...
    # labels = ...
    # model = SimpleCNN()  # Instantiate the SimpleCNN model
    # dataset = SimpleDataset(data, labels)  # Instantiate the SimpleDataset dataset
    # batch_size = 64
    #
    # evaluator = ModelEvaluator(model, dataset, batch_size)





    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 6)

    eva = ModelEvaluator(model, DatasetInterface("./dataset/datasets/train-scene/train.csv",
                            "./dataset/datasets/train-scene/train/"), 64, 34,17000,  "./dataset/datasets/train-scene/train.csv")



    metrics = eva.evaluate()
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")











    # predictions = np.array([0, 0, 1, 0, 1])
    # ground_truth = np.array([0, 1, 1, 1, 0])
    #
    # metrics = calculate_metrics(predictions, ground_truth)
    # for metric_name, metric_info in metrics.items():
    #     if isinstance(metric_info, tuple):
    #         print(f"{metric_name}: {metric_info[1]}")
    #     else:
    #         print(f"{metric_name}: {metric_info}")
    #
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


