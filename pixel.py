from evaluations import color_distance
import problem


class PixelNode:
    def __init__(
        self,
        row: int,
        col: int,
        parent,
        edge_distance: float,
        neighbor_direction: int,
        reverse_direction: bool,
    ):
        self.row = row
        self.col = col
        self.parent = parent
        self.children = []
        self.root = None
        self.distance_from_start = 0.0
        self.traversed = False
        self.neighbor_direction = neighbor_direction

        if parent is not None:
            self.distance_from_start = parent.distance_from_start + edge_distance
            if reverse_direction:
                self.neighbor_direction = int(
                    neighbor_direction + (-1) ** (neighbor_direction % 2 + 1)
                )
            else:
                self.neighbor_direction = neighbor_direction
        else:
            self.traversed = True
            self.neighbor_direction = neighbor_direction
            self.root = self

    def __lt__(self, other):
        if not isinstance(other, PixelNode):
            return NotImplemented
        return self.distance_from_start < other.distance_from_start

    def update_shortest_distance(
        self, parent, edge_distance: float, neighbor_direction: int
    ):
        if self.distance_from_start > edge_distance:
            self.parent = parent
            self.distance_from_start = edge_distance
            if neighbor_direction != 0:
                self.neighbor_direction = int(
                    neighbor_direction + (-1) ** (neighbor_direction % 2 + 1)
                )
            else:
                self.neighbor_direction = 0

    def update_grand_root(self, change_root: bool):
        orphans = []
        root = self.root
        if change_root:
            if self.parent is None:
                root = self
            else:
                root = self.parent.root
        for child in self.children:
            orphans.append(child)

        while len(orphans) > 0:
            current_orphan = orphans.pop(0)
            current_orphan.root = root
            for child in current_orphan.children:
                if child.root != root:
                    orphans.append(child)

    def get_root(self):
        family = [self]
        while family[-1].parent is not None and family[-1].parent not in family:
            family.append(family[-1].parent)
        return family[-1]

    def add_child(self, child):
        if child not in self.children:
            self.children.append(child)

    def get_updated_neighbors(self, pixel_nodes):
        neighbors_map = problem.neighbors_map
        image_height, image_width = problem.image_shape[:2]
        rgb_matrix = problem.image

        neighbors = []
        for i in range(1, 5):
            neighbor_addition = neighbors_map[i]
            neigh_row, neigh_col = (
                self.row + neighbor_addition[0],
                self.col + neighbor_addition[1],
            )
            neighbor_coords = (neigh_row, neigh_col)

            if not (0 <= neigh_row < image_height and 0 <= neigh_col < image_width):
                continue

            this_color = rgb_matrix[self.row, self.col]
            neighbor_color = rgb_matrix[neigh_row, neigh_col]
            color_dist = color_distance(this_color, neighbor_color)

            if neighbor_coords in pixel_nodes:
                neighbor = pixel_nodes[neighbor_coords]
                if not neighbor.traversed:
                    neighbor.update_shortest_distance(self, color_dist, i)
                    neighbors.append(neighbor)
            else:
                neighbor = PixelNode(neigh_row, neigh_col, self, color_dist, i, True)
                neighbors.append(neighbor)
                pixel_nodes[neighbor_coords] = neighbor

        return neighbors
