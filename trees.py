from typing import List, Dict, Tuple
from collections import defaultdict
import heapq

import numpy as np

import problem
from pixel import PixelNode


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
        neighbours = current_node.get_updated_neighbours(pixel_nodes)
        for neighbour in neighbours:
            heapq.heappush(explorables, (neighbour.distance_from_start, neighbour))
        starting_nodes.append(current_node)
        pixel_nodes[starting_coords] = current_node

    while explorables:
        _, current_node = heapq.heappop(explorables)
        if current_node.traversed:
            continue
        current_node.traversed = True
        current_node.parent.add_child(current_node)
        neighbours = current_node.get_updated_neighbours(pixel_nodes)
        for pixel_node in neighbours:
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


def get_border_pixels(
    pixel_nodes: Dict[Tuple[int, int], PixelNode]
) -> Dict[PixelNode, List[PixelNode]]:
    border_pixels = defaultdict(list)
    for row in range(problem.image_shape[0]):
        for col in range(problem.image_shape[1]):
            coords = (row, col)
            current_pixel = pixel_nodes[coords]
            if is_border_pixel(coords, pixel_nodes):
                border_pixels[current_pixel.root].append(current_pixel)

    return border_pixels


def is_border_pixel(
    coords: Tuple[int, int], pixel_nodes: Dict[Tuple[int, int], PixelNode]
) -> bool:
    root = pixel_nodes[coords].root
    for i in range(1, 5):  # 4 nearest neighbors
        neigh_coords = (
            coords[0] + problem.neighbors_map[i][0],
            coords[1] + problem.neighbors_map[i][1],
        )
        if neigh_coords in pixel_nodes:
            if pixel_nodes[neigh_coords].root != root:
                return True
    return False
