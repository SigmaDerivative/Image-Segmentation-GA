import numpy as np


import problem
from trees import (
    divide_into_trees,
    create_connected_array_with_arcs,
    decouple_array,
    generate_k_meaned_segmentation,
    segmentation_to_trees,
)


def generate_random_genome() -> np.ndarray:
    """Generate a random genome.
    Genome format is a 1D numpy array of which neighbor each pixel is connected to.

    Returns:
        np.ndarray: Numpy array filled with random values.
    """
    # initialize genome
    genome = np.random.randint(0, 5, (problem.image_shape[0], problem.image_shape[1]))

    return genome.flatten()


def generate_random_genome_from_MST() -> np.ndarray:
    """Generate a random genome from a minimum spanning tree.

    Returns:
        np.ndarray: Numpy array filled with random values.
    """
    # initialize mst
    array = create_connected_array_with_arcs(
        problem.image_shape[0], problem.image_shape[1]
    )
    decoupling_amount = np.random.randint(1, 100)
    # decouple some edges
    genome = decouple_array(array, decoupling_amount)
    return genome.flatten()


def generate_mst_genome(flat: bool = True) -> np.ndarray:
    image_height, image_width, _ = problem.image_shape

    pixel_nodes = divide_into_trees(np.random.randint(3, 8))

    genome = np.array(
        [
            pixel_nodes[(row, col)].neighbor_direction
            for row in range(image_height)
            for col in range(image_width)
        ]
    ).astype(np.uint8)
    if not flat:
        genome = genome.reshape(image_height, image_width)
    return genome


def generate_clustered_genome(flat: bool = True) -> np.ndarray:
    segmentation = generate_k_meaned_segmentation()
    genome = segmentation_to_trees(segmentation).astype(np.uint8)
    # convert from pixel_nodes to genome
    if flat:
        genome = genome.flatten()
    return genome
