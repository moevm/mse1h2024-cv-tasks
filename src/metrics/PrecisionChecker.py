from .MetricsInterface import MetricsInterface
from sklearn.metrics import precision_score
import torch

class PrecisionChecker:
    def __init__(self, name, average='macro'):
        self.name = name
        self.average = average

    def calculate_metric(self, predictions, ground_truth):
        """
        Calculate precision metric.

        Args:
            predictions: List of predicted labels.
            ground_truth: List of ground truth labels.

        Returns:
            Precision value.
        """
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu().numpy()

        return precision_score(ground_truth, predictions, average=self.average)

