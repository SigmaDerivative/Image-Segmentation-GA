from typing import List, Dict, Tuple
import heapq
from collections import deque
import random

import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage.filters import gaussian_filter

import problem
from pixel import PixelNode
from evaluations import color_distance


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


def divide_into_trees(nr_of_trees: int) -> Dict[Tuple[int, int], PixelNode]:
    image_height, image_width, _ = problem.image_shape

    pixel_nodes = {}
    explorables = []  # Priority queue (heap) instead of list
    starting_nodes = []

    # Optimize starting coordinates generation
    unique_coords = set(
        zip(
            np.random.randint(0, image_height, size=nr_of_trees * 2),
            np.random.randint(0, image_width, size=nr_of_trees * 2),
        )
    )

    for _ in range(nr_of_trees):
        while True:
            starting_coords = unique_coords.pop()
            if starting_coords not in pixel_nodes:
                break

        current_node = PixelNode(*starting_coords, None, 0, 0, True)
        neighbors = current_node.get_updated_neighbors(pixel_nodes)
        for neighbor in neighbors:
            heapq.heappush(explorables, (neighbor.distance_from_start, neighbor))
        starting_nodes.append(current_node)
        pixel_nodes[starting_coords] = current_node

    while explorables:
        _, current_node = heapq.heappop(explorables)
        if current_node.traversed:
            continue
        current_node.traversed = True
        current_node.parent.add_child(current_node)
        neighbors = current_node.get_updated_neighbors(pixel_nodes)
        for pixel_node in neighbors:
            if not pixel_node.traversed:
                heapq.heappush(
                    explorables, (pixel_node.distance_from_start, pixel_node)
                )

    for root in starting_nodes:
        root.root = root
        root.update_grand_root(False)

    return pixel_nodes


def get_crossover_indices(nr_of_sections: int) -> List[List[int]]:
    selection_sets = []
    roots = []
    pixel_nodes = divide_into_trees(nr_of_sections)
    for coords, node in pixel_nodes.items():
        if node.root not in roots:
            roots.append(node.root)
            selection_sets.append([])
        index = roots.index(node.root)
        selection_sets[index].append(coords[0] * problem.image_shape[1] + coords[1])

    return selection_sets


def generate_k_meaned_segmentation(flat: bool = False) -> np.ndarray:
    """Generate a segmentation from k means.

    Args:
        flat (bool, optional): Whether to return a flat segmentation or not. Defaults to False.
        sigma (float, optional): The standard deviation of the Gaussian filter. Defaults to 1.0.

    Returns:
        np.ndarray: segmentation.
    """
    # randomize how many clusters
    n_clusters = np.random.randint(4, 8)
    filter_sigma = np.random.uniform(2, 3)
    # generate clusters with k nearest neighbors
    # settings for faster runtime
    neigh = KMeans(n_clusters=n_clusters, n_init=3, max_iter=40).fit(
        problem.image.reshape(-1, 3)
    )
    # get cluster labels
    cluster_labels = neigh.labels_
    segmentation = cluster_labels.reshape(
        problem.image_shape[0], problem.image_shape[1]
    )

    # Apply Gaussian filter to smooth the edges
    smoothed_segmentation = gaussian_filter(
        segmentation.astype(float), sigma=filter_sigma
    )
    segmentation = np.round(smoothed_segmentation).astype(int)

    if flat:
        segmentation = segmentation.flatten()

    return segmentation


def segmentation_to_trees(segmentation: np.ndarray) -> np.ndarray[int]:
    # input is on format (height, width) with values 0- about 15
    image_height, image_width = segmentation.shape
    genome = np.zeros((image_height, image_width), dtype=int)
    visited = np.zeros_like(segmentation, dtype=bool)
    neighbors_map = {key: problem.neighbors_map[key] for key in range(1, 5)}

    def bfs(row, col, parent, init_neighbors):
        q = deque()

        for neighbor in init_neighbors:
            direction = problem.get_direction(neighbor, (row, col))
            color_dist = color_distance(
                problem.image[row, col], problem.image[neighbor[0], neighbor[1]]
            )
            neighbor_node = PixelNode(
                neighbor[0], neighbor[1], parent, color_dist, direction, False
            )
            q.append(neighbor_node)
            genome[neighbor[0], neighbor[1]] = neighbor_node.neighbor_direction
            visited[neighbor[0], neighbor[1]] = True

        while q:
            pixel_node = q.popleft()

            dirs = [1, 2, 3, 4]
            random.shuffle(dirs)

            for idx in dirs:
                direction, (di, dj) = idx, neighbors_map[idx]
                ni, nj = pixel_node.row + di, pixel_node.col + dj
                if (
                    0 <= ni < image_height
                    and 0 <= nj < image_width
                    and segmentation[ni, nj]
                    == segmentation[pixel_node.row, pixel_node.col]
                    and not visited[ni, nj]
                ):
                    new_node = PixelNode(
                        ni,
                        nj,
                        pixel_node,
                        0,
                        direction,
                        True,
                    )
                    q.append(new_node)
                    # Update pixel_nodes with the new PixelNode instance
                    genome[ni, nj] = new_node.neighbor_direction
                    visited[ni, nj] = True

    for row in range(image_height):
        for col in range(image_width):
            if not visited[row, col]:
                # create a list for the neighbors of first node in the segment
                init_neighbors = []
                # check if neighbors in segment
                for direction, (di, dj) in neighbors_map.items():
                    ni, nj = row + di, col + dj
                    if (
                        0 <= ni < image_height
                        and 0 <= nj < image_width
                        and segmentation[ni, nj] == segmentation[row, col]
                        and not visited[ni, nj]
                    ):
                        init_neighbors.append((ni, nj))
                # if only one node in segment, create an isolated node
                if len(init_neighbors) == 0:
                    root = PixelNode(row, col, None, 0, 0, True)
                    genome[row, col] = root.neighbor_direction
                    visited[row, col] = True
                else:
                    root = PixelNode(row, col, None, 0, 0, False)
                    genome[row, col] = root.neighbor_direction
                    visited[row, col] = True
                    # if more than one node in segment, create a tree
                    bfs(row, col, root, init_neighbors)

    return genome
