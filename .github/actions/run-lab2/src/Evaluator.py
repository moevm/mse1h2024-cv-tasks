import torch
import warnings
import onnxruntime
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from metrics.RecallChecker import RecallChecker
from metrics.ROCAUCChecker import ROCAUCChecker
from metrics.F1ScoreChecker import F1ScoreChecker
from metrics.AccuracyChecker import AccuracyChecker
from metrics.PrecisionChecker import PrecisionChecker
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')  # Ignore warnings
# from models.Model import model  # Import the model class
import os
from PIL import Image

# Define a class for evaluating a model
class ModelEvaluator:
    def __init__(self, model, dataset, batch_size, filename, device=None):
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
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Check if the model is PyTorch or ONNX
        if isinstance(model, torch.nn.Module):  # If model is PyTorch
            self.model = model.to(self.device)  # Move the model to the specified device
        elif isinstance(model, onnxruntime.InferenceSession):  # If model is ONNX
            self.model = model
        else:
            raise ValueError("Unsupported model type. Must be either PyTorch model or ONNX model.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(self.dataset, shuffle=False)
        self.df = pd.read_csv(filename)

    def evaluate(self):
        """
        Perform evaluation.

        Returns:
            List containing evaluation metrics.
        """
        # Check if the model is PyTorch or ONNX
        if isinstance(self.model, torch.nn.Module):  # If model is PyTorch
            self.model.eval()  # Set the model to evaluation mode

        full_metrics = []
        predictions = []
        ground_truth = []
        for images, labels in self.dataloader:
            images = images.to(self.device).float()  # Move images to the same device as the model and ensure float data type
            labels = labels.to(self.device)  # Move labels to the same device as the model
            with torch.no_grad():  # Disable gradient tracking during inference
                if isinstance(self.model, torch.nn.Module):  # If model is PyTorch
                    pred_labels = self.model(images)
                elif isinstance(self.model, onnxruntime.InferenceSession):  # If model is ONNX
                    input_name = self.model.get_inputs()[0].name  # Get the name of the input node
                    pred_labels = self.model.run(None, {input_name: images.cpu().numpy()})[0]

            model_ans = []
            for el in pred_labels:
                max_el = max(list(el))
                model_ans.append(list(el).index(max_el))

            # Move model predictions and labels to CPU before calculating metrics
            model_ans_cpu = torch.tensor(model_ans, device='cpu')
            labels_cpu = labels.cpu()

            metrics, fpr, tpr = self.calculate_metrics(model_ans_cpu, labels_cpu)
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


def resize(path="./action/datasets/lab2.2_dataset/test/test", size=224):
    """
    Resize images in a directory.

    Args:
        path: Path to the directory containing images.
        size: Size to resize the images.
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    for file in os.listdir(path):
        f_img = os.path.join(path, file)
        img = Image.open(f_img)
        img = transform(img)  # Apply transformation
        img = transforms.ToPILImage()(img)  # Convert back to PIL image
        img.save(f_img)  # Save the resized image back to the same path


