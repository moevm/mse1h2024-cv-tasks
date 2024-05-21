from .MetricsInterface import MetricsInterface
from sklearn.metrics import accuracy_score

# Creating a class named AccuracyChecker that inherits from MetricsInterface
class AccuracyChecker(MetricsInterface):
    # Constructor method that initializes the object with a name
    def __init__(self, name):
        super().__init__(name)
    
    # Method to calculate the accuracy metric
    def calculate_metric(self, predictions, ground_truth):
        return accuracy_score(ground_truth, predictions)
    
    # Method to interpret the result of the accuracy metric
    def interpret_result(self, accuracy):
        return accuracy * 100 # percentage
