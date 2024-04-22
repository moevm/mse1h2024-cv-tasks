from .MetricsInterface import MetricsInterface
from sklearn.metrics import f1_score

# Creating a class named F1ScoreChecker that inherits from MetricsInterface
class F1ScoreChecker(MetricsInterface):
    # Constructor method that initializes the object with a name and optional average parameter
    def __init__(self, name, average='macro'):
        super().__init__(name)
        self.average = average
    
    # Method to calculate the F1 score metric
    def calculate_metric(self, predictions, ground_truth):
        return f1_score(ground_truth, predictions, average=self.average)
    
    # Method to interpret the result of the F1 score metric
    def interpret_result(self, f1_score):
        return f1_score * 100 # percentage 
