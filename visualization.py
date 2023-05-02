import time
from typing import List, Dict, Tuple
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import problem


def plot_type_1(segments):
    """Plot segments, green on current image.

    Args:
        segments (np.ndarray): Segments to plot.
    """
    image = np.copy(problem.image)
    for row in range(segments.shape[0]):
        for col in range(segments.shape[1]):
            if is_border_pixel((row, col), segments):
                image[row, col] = [0, 255, 0]
    # save figure
    pil_image = Image.fromarray(image.astype(np.uint8))
    timestamp = time.time()
    os.makedirs("output/type1", exist_ok=True)
    pil_image.save(f"output/type1/segments-{timestamp}.png")


def is_border_pixel(coords: Tuple[int, int], segment: np.ndarray) -> bool:
    for i in range(1, 5):  # 4 nearest neighbors
        neigh_coords = (
            coords[0] + problem.neighbors_map[i][0],
            coords[1] + problem.neighbors_map[i][1],
        )
        # if not within bounds
        if (
            neigh_coords[0] < 0
            or neigh_coords[0] >= segment.shape[0]
            or neigh_coords[1] < 0
            or neigh_coords[1] >= segment.shape[1]
        ):
            return True
        # if within bounds
        # if different segment
        if segment[neigh_coords[0], neigh_coords[1]] != segment[coords[0], coords[1]]:
            return True
    return False


def plot_type_2(segments):
    """Plot segments, black and white border.

    Args:
        segments (np.ndarray): Segments to plot.
    """
    image = np.full(problem.image.shape, 255, dtype=np.uint8)
    for row in range(segments.shape[0]):
        for col in range(segments.shape[1]):
            if is_border_pixel((row, col), segments):
                image[row, col] = [0, 0, 0]
    # save figure
    pil_image = Image.fromarray(image.astype(np.uint8))
    timestamp = time.time()
    os.makedirs("output/type2", exist_ok=True)
    pil_image.save(f"output/type2/segments-{timestamp}.png")


def plot_type_3(segments):
    """Plot segments.

    Args:
        segments (np.ndarray): Segments to plot.
    """
    # create figure
    _, ax = plt.subplots()
    # plot image
    ax.imshow(segments)
    # set title
    ax.set_title("segments")
    # save figure
    timestamp = time.time()
    plt.savefig(f"output/segments-{timestamp}.png")
