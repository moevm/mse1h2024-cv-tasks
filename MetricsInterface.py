from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import precision_score

class MetricsInterface(ABC):
    """
    Interface class for metrics.
    """
    
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def calculate_metric(self, data):
        pass
    
    @abstractmethod
    def interpret_result(self, metric_value):
        pass

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

# Example
if __name__ == "__main__":
    
    predictions = np.array([0, 0, 1, 0, 1])
    ground_truth = np.array([0, 1, 1, 1, 0])

    precision_checker = PrecisionChecker("Precision")
    precision = precision_checker.calculate_metric(predictions, ground_truth)
    interpretation = precision_checker.interpret_result(precision)

    print(interpretation)