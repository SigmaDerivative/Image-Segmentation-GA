import numpy as np
from numba import njit

import problem


@njit
def is_connected(arr, r1, c1, r2, c2):
    if r1 == r2:
        if c1 == c2 - 1:
            return arr[r1, c1] == 1 or arr[r2, c2] == 2
        if c1 == c2 + 1:
            return arr[r1, c1] == 2 or arr[r2, c2] == 1
    if c1 == c2:
        if r1 == r2 - 1:
            return arr[r1, c1] == 4 or arr[r2, c2] == 3
        if r1 == r2 + 1:
            return arr[r1, c1] == 3 or arr[r2, c2] == 4
    return False


def dfs(arr, visited, groups, group_id, r, c):
    stack = [(r, c)]
    visited[r, c] = True
    groups[r, c] = group_id

    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    while stack:
        current_r, current_c = stack.pop()

        for direc in directions:
            r2, c2 = current_r + direc[0], current_c + direc[1]

            if 0 <= r2 < arr.shape[0] and 0 <= c2 < arr.shape[1]:
                if (
                    is_connected(arr, current_r, current_c, r2, c2)
                    and not visited[r2, c2]
                ):
                    stack.append((r2, c2))
                    visited[r2, c2] = True
                    groups[r2, c2] = group_id


def calculate_segmentation(genome):
    genome = genome.reshape(problem.image_shape[0], problem.image_shape[1])
    visited = np.zeros_like(genome, dtype=bool)
    groups = np.zeros_like(genome, dtype=int)
    group_id = 0

    for r in range(genome.shape[0]):
        for c in range(genome.shape[1]):
            if not visited[r, c]:
                dfs(genome, visited, groups, group_id, r, c)
                group_id += 1

    return groups
