from .MetricsInterface import MetricsInterface
from sklearn.metrics import f1_score

class F1ScoreChecker(MetricsInterface):
    def __init__(self, name, average='macro'):
        super().__init__(name)
        self.average = average
    
    def calculate_metric(self, predictions, ground_truth):
        return f1_score(ground_truth, predictions, average=self.average)
    
    def interpret_result(self, f1_score):
        return f1_score * 100
