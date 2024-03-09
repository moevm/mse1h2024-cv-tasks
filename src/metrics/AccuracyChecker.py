from .MetricsInterface import MetricsInterface
from sklearn.metrics import accuracy_score

class AccuracyChecker(MetricsInterface):
    def __init__(self, name):
        super().__init__(name)
    
    def calculate_metric(self, predictions, ground_truth):
        return accuracy_score(ground_truth, predictions)
    
    def interpret_result(self, accuracy):
        return f"Accuracy: {accuracy * 100:.2f}%"
