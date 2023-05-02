import numpy as np
from sklearn.cluster import KMeans

import problem
from trees import divide_into_trees, create_connected_array_with_arcs, decouple_array


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


def generate_k_meaned_segmentation(flat: bool = False) -> np.ndarray:
    """Generate a segmentation from k means.

    Args:
        flat (bool, optional): Whether to return a flat segmentation or not. Defaults to False.

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
    if not flat:
        segmentation = cluster_labels.reshape(
            problem.image_shape[0], problem.image_shape[1]
        )
    return segmentation


def generate_mst_genome(flat: bool = True) -> np.ndarray:
    image_height, image_width, _ = problem.image_shape

    pixel_nodes = divide_into_trees(np.random.randint(4, 8))

    genome = np.array(
        [
            pixel_nodes[(row, col)].neighbour_direction
            for row in range(image_height)
            for col in range(image_width)
        ]
    ).astype(np.uint8)
    if not flat:
        genome = genome.reshape(image_height, image_width)
    return genome


if __name__ == "__main__":
    import time
    from tqdm import tqdm

    before = time.time()
    for _ in tqdm(range(5)):
        genome = generate_mst_genome()

        # from visualization import plot_type_2
        # from segmentation import calculate_segmentation

        # seg = calculate_segmentation(genome)

    print(f"{time.time() - before} seconds")

    # plot_type_2(seg)
