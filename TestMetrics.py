from src.metrics import PrecisionChecker
import numpy as np

# Example
if __name__ == "__main__":
    
    predictions = np.array([0, 0, 1, 0, 1])
    ground_truth = np.array([0, 1, 1, 1, 0])

    precision_checker = PrecisionChecker("Precision")
    precision = precision_checker.calculate_metric(predictions, ground_truth)
    interpretation = precision_checker.interpret_result(precision)

    print(interpretation)