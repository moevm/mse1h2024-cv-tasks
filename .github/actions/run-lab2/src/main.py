from matplotlib import pyplot as plt
from Evaluator import ModelEvaluator, resize
from dataset import DatasetInterface
import json
import os
import torch
import subprocess

class RunMetrics():

    def __init__(self):
        self.data = json.loads(os.environ['INPUT_CORRECTPULLREQUESTS'])
        print(self.data)

    def write_message(self, index, message):
        # print(index, message)
        self.data[index]["comment"] += message
        #print(self.data[index]["comment"])
        #pass

    def write_result(self, result):
        with open(os.path.abspath(os.environ["GITHUB_OUTPUT"]), "a") as output_file:
            output_file.write(f"correctPullRequests={json.dumps(self.data)}")

    def main(self):
        for i in range(len(self.data)):
            if not self.data[i]["correct"]:
                continue
            msg = "Тестовое сообщение результата работы метрик для " + str(i) + " pr"
            self.write_message(i, msg)
        print(self.data)
        self.write_result(json.dumps(self.data))

def run_checks():
    run_metrics = RunMetrics()

    parsed_json =  json.loads(os.environ['INPUT_CORRECTPULLREQUESTS'])
    resize()

    for ind,el in enumerate(parsed_json):
        if el["lab_tag"] != "lab2":
            continue

        if not el["correct"]:
            continue
            
        for file in el["files"]:
            if "model.py" not in file["path"]:
                continue
            path = "pull-request-data/"+file["path"]
            path = path.replace("/",".")
            obj = __import__(path[:-3], fromlist=[None])
            path = path.replace(".","/")
            
            weights_file = path[:-8] + "weights.pth"
            
            if torch.cuda.is_available():
                state_dict = torch.load(weights_file)  # Load the model's state dictionary
            else:
                state_dict = torch.load(weights_file, map_location=torch.device('cpu'))

            obj.model.load_state_dict(state_dict)

            eva = ModelEvaluator(obj.model, DatasetInterface("./action/datasets/train-scene classification/train.csv",
                                                              "./action/datasets/train-scene classification/train/"),
                                  64, "./action/datasets/train-scene classification/train.csv")


            # Evaluate the model
            metrics, fpr, tpr = eva.evaluate()

            # Print evaluation metrics
            interpretation = ["Average Precision", "Average Accuracy", "Average Recall", "Average F1-score", "Average ROC-AUC"]
            for index, metric in enumerate(metrics.tolist()):
                #print(f"{interpretation[index]}: {metric}")
                #el["comment"] += f"{interpretation[index]}: {metric}\n"
                run_metrics.write_message(ind, f"{interpretation[index]}: {metric}\n")

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

    run_metrics.write_result(json.dumps(parsed_json))
    write_comments(parsed_json)

def write_comments(data):
    for i in range(len(data)):
        if (data[i]["correct"]) and data[i]["lab_tag"] == "lab2":
            command = "gh pr comment " + str(data[i]["number"]) + " --body " + "\"" + str(data[i]["comment"]) + "\""
            subprocess.run(command, shell=True, executable="/bin/bash")

if __name__ == "__main__":
    run_checks()
