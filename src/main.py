from matplotlib import pyplot as plt
from Evaluator import ModelEvaluator
from dataset import DatasetInterface
from models.Model import model  # Import the model from the Model module

if __name__ == "__main__":
    # Instantiate ModelEvaluator with the model and dataset information
    eva = ModelEvaluator(model, DatasetInterface("./dataset/datasets/train-scene/train.csv",
                                                 "./dataset/datasets/train-scene/train/"),
                         64, 8517, 8517, "./dataset/datasets/train-scene/train.csv")

    # Evaluate the model and retrieve metrics, false positive rate (fpr), and true positive rate (tpr)
    metrics, fpr, tpr = eva.evaluate()

    # Print evaluation metrics
    interpretation = ["Average Precision", "Average Accuracy", "Average Recall", "Average F1-score", "Average ROC-AUC"]
    for index, metric in enumerate(metrics.tolist()):
        print(f"{interpretation[index]}: {metric}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')  # Plot the ROC curve with a contrasting color and thicker line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # diagonal line for reference
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show() 
