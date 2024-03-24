# script for resizing images to 150x150
import PIL
import os
from PIL import Image

def resize(path="r'./dataset/datasets/train-scene/train'", size = 150):
    f = path
    for file in os.listdir(f):
        f_img = f + "/" + file
        img = Image.open(f_img)
        img = img.resize((size, size))
        img.save(f_img)