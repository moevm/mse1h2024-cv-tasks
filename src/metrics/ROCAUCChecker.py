from .MetricsInterface import MetricsInterface
from sklearn.metrics import roc_curve, auc
import numpy as np

class ROCAUCChecker(MetricsInterface):
    def __init__(self, name):
        super().__init__(name)

    def calculate_metric(self, predictions, ground_truth):
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        #
        # # Initialize containers for fpr, tpr, and roc_auc
        # fprs = []
        # tprs = []
        # roc_aucs = []
        #
        # # Compute ROC curve and ROC AUC for each class
        #
        # for class_index in range(predictions.shape[0]):
        #     fpr, tpr, _ = roc_curve(ground_truth[class_index], predictions[class_index], pos_label=class_index)
        #     roc_auc = auc(fpr, tpr)
        #     fprs.append(fpr)
        #     tprs.append(tpr)
        #     roc_aucs.append(roc_auc)
        #
        # return fprs, tprs, roc_aucs

        fpr, tpr, thresholds = roc_curve(ground_truth, predictions, pos_label=predictions.shape[0])
        return fpr, tpr, thresholds

    def interpret_result(self, roc_aucs):
        # Average ROC AUC over all classes
        avg_roc_auc = np.mean(roc_aucs)
        return f"Average ROC-AUC: {avg_roc_auc * 100:.2f}%"


# from .MetricsInterface import MetricsInterface
# from sklearn.metrics import roc_curve, auc
#
#
# class ROCAUCChecker(MetricsInterface):
#     def __init__(self, name):
#         super().__init__(name)
#
#     def calculate_metric(self, predictions, ground_truth):
#
#         fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
#         return fpr, tpr, thresholds
#
#     def interpret_result(self, roc_auc):
#         return f"ROC-AUC: {roc_auc * 100:.2f}%"