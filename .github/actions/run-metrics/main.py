from models.Model import model

import os
from PIL import Image

from src.Evaluator import ModelEvaluator
from src.dataset import DatasetInterface
import matplotlib.pyplot as plt

def resize(path="r'./datasets/train-scene/train'", size = 150):


    f = path
    for file in os.listdir(f):
        f_img = f + "/" + file
        img = Image.open(f_img)
        img = img.resize((150, 150))
        img.save(f_img)

# Example usage
if __name__ == "__main__":
    eva = ModelEvaluator(model, DatasetInterface("./dataset/datasets/train-scene/train.csv",
                                                 "./dataset/datasets/train-scene/train/"),
                         64, 8517, 8517, "./dataset/datasets/train-scene/train.csv")
    
    # Evaluate the model
    metrics, fpr, tpr = eva.evaluate()

    # Print evaluation metrics
    interpretation = ["Average Precision", "Average Accuracy", "Average Recall", "Average F1-score", "Average ROC-AUC"]
    for index, metric in enumerate(metrics.tolist()):
        print(f"{interpretation[index]}: {metric}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True) 
    plt.show()