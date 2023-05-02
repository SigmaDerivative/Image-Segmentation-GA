import numpy as np
import problem


class DisjointSetForest:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)

        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1


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


def calculate_segmentation(genome):
    rows, cols = problem.image_shape[0], problem.image_shape[1]
    genome = genome.reshape(rows, cols)
    dsf = DisjointSetForest(rows * cols)
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    for r in range(rows):
        for c in range(cols):
            for dr, dc in directions:
                r2, c2 = r + dr, c + dc
                if (
                    0 <= r2 < rows
                    and 0 <= c2 < cols
                    and is_connected(genome, r, c, r2, c2)
                ):
                    dsf.union(r * cols + c, r2 * cols + c2)

    groups = np.zeros((rows, cols), dtype=int)
    group_ids = {}

    for r in range(rows):
        for c in range(cols):
            root = dsf.find(r * cols + c)
            if root not in group_ids:
                group_ids[root] = len(group_ids)
            groups[r, c] = group_ids[root]

    return groups
