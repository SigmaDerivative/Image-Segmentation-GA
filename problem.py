"""Set up global variables for the problem."""
from PIL import Image
import numpy as np

path = "training_images/86016/Test image.jpg"
image = Image.open(path)
image = np.asarray(image)
image = image.astype(np.int32)
image_size = image.shape

neighbors_list = [
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
]
