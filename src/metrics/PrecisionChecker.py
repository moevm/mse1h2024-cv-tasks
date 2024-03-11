from .MetricsInterface import MetricsInterface
from sklearn.metrics import precision_score

class PrecisionChecker(MetricsInterface):
    """
    A class implementing precision metric checker.
    """
    
    def __init__(self, name):
        super().__init__(name)
    
    def calculate_metric(self, predictions, ground_truth):
        """
        Calculate precision metric.
        
        Parameters:
        - predictions: Predicted labels.
        - ground_truth: Ground truth labels.
        
        Returns:
        - The calculated precision.
        """
        return precision_score(ground_truth, predictions)
    
    def interpret_result(self, precision):
        """
        Interpret precision result.
        
        Parameters:
        - precision: The calculated precision.
        
        Returns:
        - A human-readable interpretation of the precision.
        """
        return f"Precision: {precision * 100:.2f}%"
