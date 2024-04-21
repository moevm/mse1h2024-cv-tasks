from Model import model

import os
from PIL import Image

from src.Evaluator import ModelEvaluator
from src.dataset import DatasetInterface

def resize(path="r'./datasets/train-scene/train'", size = 150):


    f = r'./dataset/datasets/train-scene/train'
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
    metrics = eva.evaluate()
    interpretation = ["Average Precision","Average Accuracy","Average Recall","Average F1-score"]
    for index,metric  in enumerate(metrics.tolist()):
        print(f"{interpretation[index]}: {metric}")