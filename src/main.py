from matplotlib import pyplot as plt

from Evaluator import ModelEvaluator
from Model import model
from dataset import DatasetInterface
import json



if __name__ == "__main__":
    # Instantiate ModelEvaluator

    import os


    with open('./src/action/info.json') as user_file:
        file_contents = user_file.read()


    parsed_json = json.loads(file_contents)

    for el in parsed_json:
        for file in el["files"]:
            if "main.py" in file["path"]:
                path = "action/models/"+file["path"]
                path = path.replace("/",".")
                obj = __import__(path[:-3], fromlist=[None])

                eva = ModelEvaluator(obj.model, DatasetInterface("./src/action/datasets/train-scene/train.csv",
                                                                 "./src/action/datasets/train-scene/train/"),
                                     64, "./src/action/datasets/train-scene/train.csv")

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