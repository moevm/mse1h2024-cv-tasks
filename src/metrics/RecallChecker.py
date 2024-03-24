from .MetricsInterface import MetricsInterface
from sklearn.metrics import recall_score

class RecallChecker(MetricsInterface):
    """
    A class implementing recall metric checker.
    """
    
    def __init__(self, name, average='macro'):
        super().__init__(name)
        self.average = average
    
    def calculate_metric(self, predictions, ground_truth):
        """
        Calculate recall metric.
        
        Parameters:
        - predictions: Predicted labels.
        - ground_truth: Ground truth labels.
        
        Returns:
        - The calculated recall.
        """
        return recall_score(ground_truth, predictions, average=self.average)
    
    def interpret_result(self, recall):
        """
        Interpret recall result.
        
        Parameters:
        - recall: The calculated recall.
        
        Returns:
        - A human-readable interpretation of the recall.
        """
        return recall * 100
