import numpy as np
from src.metrics.AccuracyChecker import AccuracyChecker
from src.metrics.PrecisionChecker import PrecisionChecker
from src.metrics.RecallChecker import RecallChecker
from src.metrics.F1ScoreChecker import F1ScoreChecker
from src.metrics.ROCAUCChecker import ROCAUCChecker
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Example
if __name__ == "__main__":
    
    predictions = np.array([0, 0, 1, 0, 1])
    ground_truth = np.array([0, 1, 1, 1, 0])

    # Lists to store metric values
    precision_values = []
    accuracy_values = []
    recall_values = []
    f1_score_values = []
    roc_auc_values = []

    # Precision
    precision_checker = PrecisionChecker("Precision")
    precision = precision_checker.calculate_metric(predictions, ground_truth)
    precision_values.append(precision)
    precision_interpretation = precision_checker.interpret_result(precision)
    print(precision_interpretation)

    # Accuracy
    accuracy_checker = AccuracyChecker("Accuracy")
    accuracy = accuracy_checker.calculate_metric(predictions, ground_truth)
    accuracy_values.append(accuracy)
    accuracy_interpretation = accuracy_checker.interpret_result(accuracy)
    print(accuracy_interpretation)

    # Recall
    recall_checker = RecallChecker("Recall")
    recall = recall_checker.calculate_metric(predictions, ground_truth)
    recall_values.append(recall)
    recall_interpretation = recall_checker.interpret_result(recall)
    print(recall_interpretation)

    # F1-score
    f1_score_checker = F1ScoreChecker("F1-score")
    f1_score = f1_score_checker.calculate_metric(predictions, ground_truth)
    f1_score_values.append(f1_score)
    f1_score_interpretation = f1_score_checker.interpret_result(f1_score)
    print(f1_score_interpretation)

    # ROC-AUC
    roc_auc_checker = ROCAUCChecker("ROC-AUC")
    fpr, tpr, thresholds = roc_auc_checker.calculate_metric(predictions, ground_truth)
    roc_auc = auc(fpr, tpr)
    roc_auc_values.append(roc_auc)
    roc_auc_interpretation = roc_auc_checker.interpret_result(roc_auc)
    print(roc_auc_interpretation)

    # Calculate averages
    average_precision = np.mean(precision_values)
    average_accuracy = np.mean(accuracy_values)
    average_recall = np.mean(recall_values)
    average_f1_score = np.mean(f1_score_values)
    average_roc_auc = np.mean(roc_auc_values)

    # Print average values
    print("Average Precision:", average_precision)
    print("Average Accuracy:", average_accuracy)
    print("Average Recall:", average_recall)
    print("Average F1-score:", average_f1_score)
    print("Average ROC-AUC:", average_roc_auc)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()
