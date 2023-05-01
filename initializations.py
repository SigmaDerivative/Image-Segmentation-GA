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
    genome = np.random.randint(0, 5, (problem.image_shape[0], problem.image_shape[1]))

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


def generate_mst_genome(problem):
    rgb_matrix = problem.image
    image_height, image_width, _ = problem.image_shape
    neighbors_map = problem.neighbors_map

    visited = np.zeros(image_height * image_width, dtype=bool)
    mst = np.full((image_height, image_width), -1, dtype=int)
    explorable = {
        i * image_width + j for i in range(image_height) for j in range(image_width)
    }

    while explorable:
        has_neighbor = True
        current_index = explorable.pop()
        current_node = np.array(
            [current_index // image_width, current_index % image_width]
        )
        visited[current_index] = True

        while has_neighbor:
            neighbors = [
                (i, current_node + neighbors_map[i])
                for i in range(1, 5)
                if 0 <= current_node[0] + neighbors_map[i][0] < image_height
                and 0 <= current_node[1] + neighbors_map[i][1] < image_width
                and not visited[
                    (current_node[0] + neighbors_map[i][0]) * image_width
                    + current_node[1]
                    + neighbors_map[i][1]
                ]
            ]

            if neighbors:
                best_neighbor, next_node = min(
                    neighbors,
                    key=lambda x: np.sum(
                        (rgb_matrix[tuple(current_node)] - rgb_matrix[tuple(x[1])]) ** 2
                    ),
                )
                mst[current_node[0], current_node[1]] = best_neighbor
                current_node = next_node
                current_index = current_node[0] * image_width + current_node[1]
                visited[current_index] = True
                explorable.discard(current_index)
            else:
                visited[current_index] = True
                mst[current_node[0], current_node[1]] = 0
                has_neighbor = False

    return mst


if __name__ == "__main__":
    genome = generate_mst_genome(problem)

    from visualization import plot_type_2
    from segmentation import calculate_segmentation

    seg = calculate_segmentation(genome)

    plot_type_2(seg)
