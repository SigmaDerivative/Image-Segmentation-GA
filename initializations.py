import numpy as np
from sklearn.cluster import KMeans

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
    # genome[0, :] = np.random.choice([0, 1, 2, 3], size=problem.image_size[1])
    # genome[-1, :] = np.random.choice([0, 2, 3, 4], size=problem.image_size[1])
    # genome[:, 0] = np.random.choice([0, 1, 2, 3], size=problem.image_size[0])
    # genome[:, -1] = np.random.choice([0, 1, 3, 4], size=problem.image_size[0])
    # overwrite corners with legal values
    # genome[0, 0] = np.random.choice([0, 1, 2])
    # genome[0, -1] = np.random.choice([0, 1, 4])
    # genome[-1, 0] = np.random.choice([0, 2, 3])
    # genome[-1, -1] = np.random.choice([0, 3, 4])

    return genome


def create_connected_array_with_arcs(rows, cols):
    # Create an empty NumPy array
    arr = np.zeros((rows, cols), dtype=int)

    def get_neighbors(i, j):
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j, 3))
        if j > 0:
            neighbors.append((i, j - 1, 4))
        if i < rows - 1:
            neighbors.append((i + 1, j, 1))
        if j < cols - 1:
            neighbors.append((i, j + 1, 2))
        return neighbors

    visited = set()
    stack = [(0, 0)]

    while stack:
        i, j = stack.pop()
        if (i, j) not in visited:
            visited.add((i, j))
            neighbors = get_neighbors(i, j)
            np.random.shuffle(neighbors)
            for ni, nj, direction in neighbors:
                if (ni, nj) not in visited:
                    arr[i, j] = direction
                    stack.append((ni, nj))
                    break

    # Ensure that the entire array is connected
    for i in range(rows):
        for j in range(cols):
            if arr[i, j] == 0:
                neighbors = get_neighbors(i, j)
                np.random.shuffle(neighbors)
                for ni, nj, direction in neighbors:
                    if arr[ni, nj] != 0:
                        arr[i, j] = direction
                        break

    return arr


def decouple_array(array, number_to_decouple):
    for _ in range(number_to_decouple):
        # Get random row and column
        row = np.random.randint(0, array.shape[0] - 1)
        col = np.random.randint(0, array.shape[1] - 1)

        # Get the direction of the neighbor
        direction = array[row, col]

        # Decouple the neighbor
        if direction == 1:
            array[row + 1, col] = 0
        elif direction == 2:
            array[row, col + 1] = 0
        elif direction == 3:
            array[row - 1, col] = 0
        elif direction == 4:
            array[row, col - 1] = 0

    return array


def generate_random_genome_from_MST() -> np.ndarray:
    """Generate a random genome from a minimum spanning tree.

    Returns:
        np.ndarray: Numpy array filled with random values.
    """
    # initialize mst
    array = create_connected_array_with_arcs(
        problem.image_size[0], problem.image_size[1]
    )
    decoupling_amount = np.random.randint(1, 100)
    # decouple some edges
    genome = decouple_array(array, decoupling_amount)
    return genome.flatten()


def generate_k_meaned_segmentation() -> np.ndarray:
    """Generate a segmentation from k means.

    Returns:
        np.ndarray: segmentation.
    """
    # randomize how many clusters
    n_clusters = np.random.randint(3, 20)
    # generate clusters with k nearest neighbors
    # settings for faster runtime
    neigh = KMeans(n_clusters=n_clusters, n_init=3, max_iter=40).fit(
        problem.image.reshape(-1, 3)
    )
    # get cluster labels
    cluster_labels = neigh.labels_
    segmentation = cluster_labels.reshape(problem.image_size[0], problem.image_size[1])
    return segmentation


def generate_k_meaned_segmentation_flat() -> np.ndarray:
    """Generate a segmentation from k means.

    Returns:
        np.ndarray: segmentation.
    """
    # randomize how many clusters
    n_clusters = np.random.randint(3, 20)
    # generate clusters with k nearest neighbors
    # settings for faster runtime
    neigh = KMeans(n_clusters=n_clusters, n_init=3, max_iter=40).fit(
        problem.image.reshape(-1, 3)
    )
    # get cluster labels
    segmentation = neigh.labels_
    return segmentation


def generate_random_population(population_size: int) -> list:
    """Generate a random population.

    Args:
        population_size (int): Size of the population.

    Returns:
        list: List of random genomes.
    """

    return [generate_random_genome() for _ in range(population_size)]
