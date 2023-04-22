import numpy as np

import problem


def generate_random_genome() -> np.ndarray:
    """Generate a random genome.
    Genome format is a 2D array of which neighbor each pixel is connected to.
    The values are:
    0: no connection
    1: connection to the pixel below
    2: connection to the pixel to the right
    3: connection to the pixel above
    4: connection to the pixel to the left

    Returns:
        np.ndarray: Numpy array filled with random values.
    """
    # initialize genome
    genome = np.random.randint(0, 5, (problem.image_size[0], problem.image_size[1]))
    # overwrite edges with legal values
    genome[0, :] = np.random.choice([0, 1, 2, 3], size=problem.image_size[1])
    genome[-1, :] = np.random.choice([0, 2, 3, 4], size=problem.image_size[1])
    genome[:, 0] = np.random.choice([0, 1, 2, 3], size=problem.image_size[0])
    genome[:, -1] = np.random.choice([0, 1, 3, 4], size=problem.image_size[0])
    # overwrite corners with legal values
    genome[0, 0] = np.random.choice([0, 1, 2])
    genome[0, -1] = np.random.choice([0, 1, 4])
    genome[-1, 0] = np.random.choice([0, 2, 3])
    genome[-1, -1] = np.random.choice([0, 3, 4])

    return genome


def generate_random_population(population_size: int) -> list:
    """Generate a random population.

    Args:
        population_size (int): Size of the population.

    Returns:
        list: List of random genomes.
    """
    return [generate_random_genome() for _ in range(population_size)]
