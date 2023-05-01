from typing import List, Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans

import problem
from pixel import PixelNode
from evaluations import color_distance


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


def get_updated_neighbours(pixel_node, pixel_nodes, problem):
    neighbors_map = problem.neighbors_map
    image_height, image_width = problem.image_shape[:2]
    rgb_matrix = problem.image

    neighbours = []
    for i in range(1, 5):
        neighbour_addition = neighbors_map[i]
        neigh_row, neigh_col = (
            pixel_node.row + neighbour_addition[0],
            pixel_node.col + neighbour_addition[1],
        )
        neighbour_coords = (neigh_row, neigh_col)

        if not (0 <= neigh_row < image_height and 0 <= neigh_col < image_width):
            continue

        this_color = rgb_matrix[pixel_node.row, pixel_node.col]
        neighbour_color = rgb_matrix[neigh_row, neigh_col]
        color_dist = color_distance(this_color, neighbour_color)

        if neighbour_coords in pixel_nodes:
            neighbour = pixel_nodes[neighbour_coords]
            if not neighbour.traversed:
                neighbour.update_shortest_distance(pixel_node, color_dist, i)
                neighbours.append(neighbour)
        else:
            neighbour = PixelNode(neigh_row, neigh_col, pixel_node, color_dist, i, True)
            neighbours.append(neighbour)
            pixel_nodes[neighbour_coords] = neighbour

    return neighbours


def generate_mst_genome(problem):
    image_height, image_width, _ = problem.image_shape

    pixel_nodes = {}
    explorables = []
    starting_nodes = []

    nr_of_trees = np.random.randint(2, 10)

    for _ in range(nr_of_trees):
        while True:
            starting_coords = np.random.randint(0, image_height), np.random.randint(
                0, image_width
            )
            if starting_coords not in pixel_nodes and not any(
                starting_coords == (explorable.row, explorable.col)
                for explorable in explorables
            ):
                break

        current_node = PixelNode(*starting_coords, None, 0, 0, True)
        neighbours = get_updated_neighbours(current_node, pixel_nodes, problem)
        explorables.extend(neighbours)
        starting_nodes.append(current_node)
        pixel_nodes[starting_coords] = current_node

    while explorables:
        current_node = min(
            explorables, key=lambda pixel_node: pixel_node.distance_from_start
        )
        current_node.traversed = True
        current_node.parent.add_child(current_node)
        explorables.remove(current_node)
        neighbours = get_updated_neighbours(current_node, pixel_nodes, problem)
        for pixel_node in neighbours:
            if pixel_node not in explorables and not pixel_node.traversed:
                explorables.append(pixel_node)

    for root in starting_nodes:
        root.segment_root = root
        root.update_all_segment_root(False)

    genome = (
        np.array(
            [
                pixel_nodes[(row, col)].neighbour_direction
                for row in range(image_height)
                for col in range(image_width)
            ]
        )
        .reshape(image_height, image_width)
        .astype(np.uint8)
    )
    return genome


if __name__ == "__main__":
    genome = generate_mst_genome(problem)

    from visualization import plot_type_2
    from segmentation import calculate_segmentation

    seg = calculate_segmentation(genome)

    plot_type_2(seg)
