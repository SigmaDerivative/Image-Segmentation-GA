"""Set up global variables for the problem."""
from PIL import Image
import numpy as np

path = "training_images/86016/Test image.jpg"
image = Image.open(path)
image = np.asarray(image)
image = image.astype(np.int32)
image_shape = image.shape

neighbors_list = [
    (0, 0),
    (0, 1),
    (0, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (1, 1),
    (-1, -1),
    (1, -1),
]
neighbors_map = {
    0: np.array([0, 0]),
    1: np.array([0, 1]),
    2: np.array([0, -1]),
    3: np.array([-1, 0]),
    4: np.array([1, 0]),
    5: np.array([1, -1]),
    6: np.array([1, 1]),
    7: np.array([-1, 1]),
    8: np.array([-1, -1]),
}
