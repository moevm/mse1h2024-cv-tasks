import unittest
import numpy as np
from src.metrics.AccuracyChecker import AccuracyChecker
from src.metrics.PrecisionChecker import PrecisionChecker
from src.metrics.RecallChecker import RecallChecker
from src.metrics.F1ScoreChecker import F1ScoreChecker
from src.metrics.ROCAUCChecker import ROCAUCChecker

from sklearn.metrics import auc


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.predictions = np.array([0, 0, 1, 0, 1])
        self.ground_truth = np.array([0, 1, 1, 1, 0])
        self.ground_truth_incorrect = np.array([1, 1, 0, 1, 0])

        self.exp = 0.0000001

    def test_recision(self):
        checker = PrecisionChecker("TestPrecision")

        result = checker.calculate_metric(self.predictions, self.predictions)
        self.assertEqual(result, 1)
        self.assertEqual(checker.interpret_result(result), "Precision: 100.00%")

        result = checker.calculate_metric(self.predictions, self.ground_truth_incorrect)
        self.assertEqual(result, 0)
        self.assertEqual(checker.interpret_result(result), "Precision: 0.00%")

        result = checker.calculate_metric(self.predictions, self.ground_truth)
        self.assertEqual(result, 0.5)
        self.assertEqual(checker.interpret_result(result), "Precision: 50.00%")

    def test_accuracy(self):
        checker = AccuracyChecker("TestAccuracy")

        result = checker.calculate_metric(self.predictions, self.predictions)
        self.assertEqual(result, 1)
        self.assertEqual(checker.interpret_result(result), "Accuracy: 100.00%")

        result = checker.calculate_metric(self.predictions, self.ground_truth_incorrect)
        self.assertEqual(result, 0)
        self.assertEqual(checker.interpret_result(result), "Accuracy: 0.00%")

        result = checker.calculate_metric(self.predictions, self.ground_truth)
        self.assertEqual(result, 0.4)
        self.assertEqual(checker.interpret_result(result), "Accuracy: 40.00%")

    def test_recall(self):
        checker = RecallChecker("TestRecall")

        result = checker.calculate_metric(self.predictions, self.predictions)
        self.assertEqual(result, 1)
        self.assertEqual(checker.interpret_result(result), "Recall: 100.00%")

        result = checker.calculate_metric(self.predictions, self.ground_truth_incorrect)
        self.assertEqual(result, 0)
        self.assertEqual(checker.interpret_result(result), "Recall: 0.00%")

        result = checker.calculate_metric(self.predictions, self.ground_truth)
        self.assertTrue(abs(result - 1 / 3) < self.exp)
        self.assertEqual(checker.interpret_result(result), "Recall: 33.33%")

    def test_f1(self):
        checker = F1ScoreChecker("TestF1")

        result = checker.calculate_metric(self.predictions, self.predictions)
        self.assertEqual(result, 1)
        self.assertEqual(checker.interpret_result(result), "F1-score: 100.00%")

        result = checker.calculate_metric(self.predictions, self.ground_truth_incorrect)
        self.assertEqual(result, 0)
        self.assertEqual(checker.interpret_result(result), "F1-score: 0.00%")

        result = checker.calculate_metric(self.predictions, self.ground_truth)
        self.assertEqual(result, 0.4)
        self.assertEqual(checker.interpret_result(result), "F1-score: 40.00%")

    def test_roc_auc(self):
        checker = ROCAUCChecker("TestROCAUC")

        fpr, tpr, thresholds = checker.calculate_metric(self.predictions, self.predictions)
        result = auc(fpr, tpr)
        self.assertEqual(result, 1)
        self.assertEqual(checker.interpret_result(result), "ROC-AUC: 100.00%")

        fpr, tpr, thresholds = checker.calculate_metric(self.predictions, self.ground_truth_incorrect)
        result = auc(fpr, tpr)
        self.assertEqual(result, 0)
        self.assertEqual(checker.interpret_result(result), "ROC-AUC: 0.00%")

        fpr, tpr, thresholds = checker.calculate_metric(self.predictions, self.ground_truth)
        result = auc(fpr, tpr)
        self.assertTrue(abs(result - 5 / 12) < self.exp)
        self.assertEqual(checker.interpret_result(result), "ROC-AUC: 41.67%")


# Example
if __name__ == "__main__":
    unittest.main()
