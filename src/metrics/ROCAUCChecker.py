from .MetricsInterface import MetricsInterface
from sklearn.metrics import roc_curve, auc

class ROCAUCChecker(MetricsInterface):
    def __init__(self, name):
        super().__init__(name)
    
    def calculate_metric(self, predictions, ground_truth):
        fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
        return fpr, tpr, thresholds
    
    def interpret_result(self, roc_auc):
        return f"ROC-AUC: {roc_auc * 100:.2f}%"
