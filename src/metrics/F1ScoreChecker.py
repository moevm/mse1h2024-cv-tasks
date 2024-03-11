from .MetricsInterface import MetricsInterface
from sklearn.metrics import f1_score

class F1ScoreChecker(MetricsInterface):
    def __init__(self, name):
        super().__init__(name)
    
    def calculate_metric(self, predictions, ground_truth):
        return f1_score(ground_truth, predictions)
    
    def interpret_result(self, f1_score):
        return f"F1-score: {f1_score * 100:.2f}%"
