import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from metrics.RecallChecker import RecallChecker
from metrics.ROCAUCChecker import ROCAUCChecker
from metrics.F1ScoreChecker import F1ScoreChecker
from metrics.AccuracyChecker import AccuracyChecker
from metrics.PrecisionChecker import PrecisionChecker
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings('ignore')  # Ignore warnings
# from models.Model import model  # Import the model class
import os
from PIL import Image

# Define a class for evaluating a model
class ModelEvaluator:
    def __init__(self, model, dataset, batch_size, filename):
        """
        Initialize the ModelEvaluator.

        Args:
            model_file: Path to the model file.
            dataset: The dataset to be used for evaluation.
            batch_size: Batch size for DataLoader.
            predictions_len: Length of predictions for splitting dataset.
            ground_truth_len: Length of ground truth for splitting dataset.
            filename: CSV filename.
        """
        # Load the model from the specified file
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU
        self.model = model.to(self.device)  # Move the model to the specified device
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.df = pd.read_csv(filename)

    def evaluate(self):
        """
        Perform evaluation.
    
        Returns:
            List containing evaluation metrics.
        """
        # Set the model to evaluation mode
        self.model.eval()
    
        full_metrics = []
        predictions = []
        ground_truth = []
        for images, labels in self.dataloader:
            images = images.to(self.device).float()  # Move images to the same device as the model and ensure float data type
            labels = labels.to(self.device)  # Move labels to the same device as the model
            with torch.no_grad():  # Disable gradient tracking during inference
                pred_labels = self.model(images)
            
            model_ans = []
            for el in pred_labels:
                max_el = max(list(el))
                model_ans.append(list(el).index(max_el))
            
            # Move model predictions and labels to CPU before calculating metrics
            model_ans_cpu = torch.tensor(model_ans, device='cpu')
            labels_cpu = labels.cpu()
    
            metrics, fpr, tpr= self.calculate_metrics(model_ans_cpu, labels_cpu)
            full_metrics.append(metrics)
            predictions.extend(model_ans)
            ground_truth.extend(labels_cpu)
        
        return np.mean(full_metrics, axis=0), fpr, tpr

    def calculate_metrics(self, predictions, ground_truth):
        """
        Calculate evaluation metrics.

        Args:
            predictions: List of predicted labels.
            ground_truth: List of ground truth labels.

        Returns:
            List containing evaluation metrics.
        """
        # Convert predictions to numpy array
        predictions = np.array(predictions)

        # Move ground truth tensor to CPU and convert to numpy array
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu().detach().numpy()
        else:
            ground_truth = np.array(ground_truth)

        # Precision
        precision_checker = PrecisionChecker("Precision", average='macro')
        precision = precision_checker.calculate_metric(predictions, ground_truth)

        # Accuracy
        accuracy_checker = AccuracyChecker("Accuracy")
        accuracy = accuracy_checker.calculate_metric(predictions, ground_truth)

        # Recall
        recall_checker = RecallChecker("Recall", average='macro')
        recall = recall_checker.calculate_metric(predictions, ground_truth)

        # F1-score
        f1_score_checker = F1ScoreChecker("F1-score", average='macro')
        f1_score = f1_score_checker.calculate_metric(predictions, ground_truth)

        # ROC-AUC for each class
        roc_auc_checker = ROCAUCChecker("ROC-AUC")
        fpr, tpr, roc_auc = roc_auc_checker.calculate_metric(predictions, ground_truth)
        
        # Ensure all metrics have fixed lengths
        fixed_metrics = [precision, accuracy, recall, f1_score, roc_auc]

        return fixed_metrics, fpr, tpr

def resize(path="./action/cw2_dataset/cw2.2_dataset/images", size=150):
    """
    Resize images in a directory.

    Args:
        path: Path to the directory containing images.
        size: Size to resize the images.
    """
    for file in os.listdir(path):
        f_img = os.path.join(path, file)
        img = Image.open(f_img)
        img = img.resize((size, size))  # Resize the image
        img.save(f_img)  # Save the resized image back to the same path

