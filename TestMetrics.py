from src.metrics.AccuracyChecker import AccuracyChecker
from src.metrics.PrecisionChecker import PrecisionChecker
from src.metrics.RecallChecker import RecallChecker
from src.metrics.F1ScoreChecker import F1ScoreChecker
import numpy as np

# Example
if __name__ == "__main__":
    
    predictions = np.array([0, 0, 1, 0, 1])
    ground_truth = np.array([0, 1, 1, 1, 0])

    precision_checker = PrecisionChecker("Precision")
    precision = precision_checker.calculate_metric(predictions, ground_truth)
    precision_interpretation = precision_checker.interpret_result(precision)
    print(precision_interpretation)

    accuracy_checker = AccuracyChecker("Accuracy")
    accuracy = accuracy_checker.calculate_metric(predictions, ground_truth)
    accuracy_interpretation = accuracy_checker.interpret_result(accuracy)
    print(accuracy_interpretation)

    recall_checker = RecallChecker("Recall")
    recall = recall_checker.calculate_metric(predictions, ground_truth)
    recall_interpretation = recall_checker.interpret_result(recall)
    print(recall_interpretation)

    f1_score_checker = F1ScoreChecker("F1-score")
    f1_score = f1_score_checker.calculate_metric(predictions, ground_truth)
    f1_score_interpretation = f1_score_checker.interpret_result(f1_score)
    print(f1_score_interpretation)