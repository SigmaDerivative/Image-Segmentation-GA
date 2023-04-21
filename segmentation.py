import numpy as np


def is_connected(arr, r1, c1, r2, c2):
    if r1 == r2 and c1 == c2 - 1 and (arr[r1, c1] == 2 or arr[r2, c2] == 4):
        return True
    elif r1 == r2 and c1 == c2 + 1 and (arr[r1, c1] == 4 or arr[r2, c2] == 2):
        return True
    elif r1 == r2 - 1 and c1 == c2 and (arr[r1, c1] == 1 or arr[r2, c2] == 3):
        return True
    elif r1 == r2 + 1 and c1 == c2 and (arr[r1, c1] == 3 or arr[r2, c2] == 1):
        return True
    return False


def dfs(arr, visited, groups, group_id, r, c):
    visited[r, c] = True
    groups[r, c] = group_id

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for _, direc in enumerate(directions):
        r2, c2 = r + direc[0], c + direc[1]
        if 0 <= r2 < arr.shape[0] and 0 <= c2 < arr.shape[1]:
            if is_connected(arr, r, c, r2, c2) and not visited[r2, c2]:
                dfs(arr, visited, groups, group_id, r2, c2)


def segmentation(arr):
    visited = np.zeros_like(arr, dtype=bool)
    groups = np.zeros_like(arr, dtype=int)
    group_id = 0

    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if not visited[r, c]:
                dfs(arr, visited, groups, group_id, r, c)
                group_id += 1

    return groups
