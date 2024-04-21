from models.Model import model
import os
from PIL import Image
import json
import subprocess
#from src.Evaluator import ModelEvaluator
#from src.dataset import DatasetInterface


'''def resize(path="r'./datasets/train-scene/train'", size = 150):

    f = path
    for file in os.listdir(f):
        f_img = f + "/" + file
        img = Image.open(f_img)
        img = img.resize((150, 150))
        img.save(f_img)'''
class RunMetrics():

    def __init__(self):
        self.data = json.loads(os.environ['INPUT_CORRECTPULLREQUESTS'])
        print(self.data)
    
    def write_message(self, index, message):
        #print(index, message)   
        self.data[index]["comment"] += message
        print(self.data[index]["comment"])
        pass
    def write_result(self, result):
        with open(os.path.abspath(os.environ["GITHUB_OUTPUT"]), "a") as output_file:
            output_file.write(f"correctPullRequests={result}")
    def main(self):
        for i in range(len(self.data)):
            if not self.data[i]["correct"]:
                continue
            msg = "Тестовое сообщение результата работы метрик для "+str(i)+ " pr"
            self.write_message(i, msg)
        print(self.data)
        self.write_result(json.dumps(self.data))
# Example usage
if __name__ == "__main__":
    run_merics = RunMetrics()
    run_merics.main()
    '''eva = ModelEvaluator(model, DatasetInterface("./dataset/datasets/train-scene/train.csv",
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
