from .MetricsInterface import MetricsInterface
from sklearn.metrics import roc_curve, auc
import numpy as np

class ROCAUCChecker(MetricsInterface):
    def __init__(self, name):
        # Calling the constructor of the parent class (MetricsInterface)
        super().__init__(name)

    def calculate_metric(self, predictions, ground_truth):
        fpr, tpr, thresholds = roc_curve(ground_truth, predictions, pos_label=1)  # Assuming binary classification
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def interpret_result(self, roc_auc):
        return f"ROC-AUC: {roc_auc * 100:.2f}%" # percentage 
