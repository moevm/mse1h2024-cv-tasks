import numpy as np
from src.metrics.AccuracyChecker import AccuracyChecker
from src.metrics.PrecisionChecker import PrecisionChecker
from src.metrics.RecallChecker import RecallChecker
from src.metrics.F1ScoreChecker import F1ScoreChecker
from src.metrics.ROCAUCChecker import ROCAUCChecker
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def calculate_metrics(predictions, ground_truth):
    # Initialize containers for metric values and interpretations
    metric_values = {}
    metric_interpretations = {}

    # Precision
    precision_checker = PrecisionChecker("Precision")
    precision = precision_checker.calculate_metric(predictions, ground_truth)
    precision_interpretation = precision_checker.interpret_result(precision)
    metric_values["Precision"] = precision
    metric_interpretations["Precision"] = precision_interpretation

    # Accuracy
    accuracy_checker = AccuracyChecker("Accuracy")
    accuracy = accuracy_checker.calculate_metric(predictions, ground_truth)
    accuracy_interpretation = accuracy_checker.interpret_result(accuracy)
    metric_values["Accuracy"] = accuracy
    metric_interpretations["Accuracy"] = accuracy_interpretation

    # Recall
    recall_checker = RecallChecker("Recall")
    recall = recall_checker.calculate_metric(predictions, ground_truth)
    recall_interpretation = recall_checker.interpret_result(recall)
    metric_values["Recall"] = recall
    metric_interpretations["Recall"] = recall_interpretation

    # F1-score
    f1_score_checker = F1ScoreChecker("F1-score")
    f1_score = f1_score_checker.calculate_metric(predictions, ground_truth)
    f1_score_interpretation = f1_score_checker.interpret_result(f1_score)
    metric_values["F1-score"] = f1_score
    metric_interpretations["F1-score"] = f1_score_interpretation

    # ROC-AUC
    roc_auc_checker = ROCAUCChecker("ROC-AUC")
    fpr, tpr, thresholds = roc_auc_checker.calculate_metric(predictions, ground_truth)
    roc_auc = auc(fpr, tpr)
    roc_auc_interpretation = roc_auc_checker.interpret_result(roc_auc)
    metric_values["ROC-AUC"] = roc_auc
    metric_interpretations["ROC-AUC"] = roc_auc_interpretation

    # Calculate averages
    average_precision = np.mean(list(metric_values.values()))
    average_accuracy = np.mean(list(metric_values.values()))
    average_recall = np.mean(list(metric_values.values()))
    average_f1_score = np.mean(list(metric_values.values()))
    average_roc_auc = np.mean(list(metric_values.values()))

    # Return metrics and interpretations as tuples
    metrics = {
        **metric_values,
        **metric_interpretations,
        "Average Precision": average_precision,
        "Average Accuracy": average_accuracy,
        "Average Recall": average_recall,
        "Average F1-score": average_f1_score,
        "Average ROC-AUC": average_roc_auc,
        "ROC Curve": (fpr, tpr, roc_auc)
    }

    return metrics

# Example usage
if __name__ == "__main__":
    predictions = np.array([0, 0, 1, 0, 1])
    ground_truth = np.array([0, 1, 1, 1, 0])
    
    metrics = calculate_metrics(predictions, ground_truth)
    for metric_name, metric_info in metrics.items():
        print(f"{metric_name}: {metric_info}")

    # Plot ROC curve
    fpr, tpr, roc_auc = metrics["ROC Curve"]
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()
