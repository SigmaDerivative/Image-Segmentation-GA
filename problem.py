"""Set up global variables for the problem."""
from PIL import Image
import numpy as np

image_id = "353013"
path = f"training_images\\{image_id}\\Test image.jpg"
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


def get_direction(coords1, coords2):
    """Get the direction from coords1 to coords2.

    Args:
        coords1 (Tuple[int, int]): Coordinates of the first pixel.
        coords2 (Tuple[int, int]): Coordinates of the second pixel.

    Returns:
        int: Direction from coords1 to coords2.
    """
    direction = np.array(coords2) - np.array(coords1)
    for key, value in neighbors_map.items():
        if np.array_equal(value, direction):
            return key

    raise ValueError(f"Could not find direction from {coords1} to {coords2}.")
