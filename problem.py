"""Set up global variables for the problem."""
from PIL import Image
import numpy as np

path = "training_images/86016/Test image.jpg"
image = Image.open(path)
image = np.asarray(image)
image = image.astype(np.int32)
image_size = image.shape
