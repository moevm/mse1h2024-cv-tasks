from src.metrics.AccuracyChecker import AccuracyChecker
from src.metrics.PrecisionChecker import PrecisionChecker
import numpy as np

# Example
if __name__ == "__main__":
    
    predictions = np.array([0, 0, 1, 0, 1])
    ground_truth = np.array([0, 1, 1, 1, 0])

    precision_checker = PrecisionChecker("Precision")
    precision = precision_checker.calculate_metric(predictions, ground_truth)
    precision_interpretation = precision_checker.interpret_result(precision)
    print("Precision:", precision_interpretation)

    accuracy_checker = AccuracyChecker("Accuracy")
    accuracy = accuracy_checker.calculate_metric(predictions, ground_truth)
    accuracy_interpretation = accuracy_checker.interpret_result(accuracy)
    print("Accuracy:", accuracy_interpretation)
