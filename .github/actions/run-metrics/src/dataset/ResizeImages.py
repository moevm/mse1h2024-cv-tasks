import PIL
import os
from PIL import Image

def resize(path="./dataset/datasets/train-scene/train", size=150):
    """
    Resize images to a specified size.
    
    Args:
    path (str): Path to the directory containing images.
    size (int): Desired size for the images (both width and height).
    """
    f = path
    for file in os.listdir(f):
        f_img = f + "/" + file
        img = Image.open(f_img)
        img = img.resize((size, size))
        img.save(f_img)
