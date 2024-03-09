from .MetricsInterface import MetricsInterface
from sklearn.metrics import recall_score

class RecallChecker(MetricsInterface):
    def __init__(self, name):
        super().__init__(name)
    
    def calculate_metric(self, predictions, ground_truth):
        return recall_score(ground_truth, predictions)
    
    def interpret_result(self, recall):
        return f"Recall: {recall * 100:.2f}%"
