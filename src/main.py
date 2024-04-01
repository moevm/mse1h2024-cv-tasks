from matplotlib import pyplot as plt

from Evaluator import ModelEvaluator
from Model import model
from dataset import DatasetInterface

if __name__ == "__main__":
    # Instantiate ModelEvaluator
    eva = ModelEvaluator(model, DatasetInterface("./src/dataset/datasets/train-scene/train.csv",
                                                 "./src/dataset/datasets/train-scene/train/"),
                         64, 8517, 8517, "./src/dataset/datasets/train-scene/train.csv")

    # Evaluate the model
    metrics, fpr, tpr = eva.evaluate()

    # Print evaluation metrics
    interpretation = ["Average Precision", "Average Accuracy", "Average Recall", "Average F1-score", "Average ROC-AUC"]
    for index, metric in enumerate(metrics.tolist()):
        print(f"{interpretation[index]}: {metric}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')  # Use a contrasting color and thicker line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)  # Add gridlines
    plt.show()
