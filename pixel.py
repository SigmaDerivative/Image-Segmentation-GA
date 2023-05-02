from evaluations import color_distance
import problem


class PixelNode:
    def __init__(
        self,
        row: int,
        col: int,
        parent,
        edge_distance: float,
        neighbour_direction: int,
        reverse_direction: bool,
    ):
        self.row = row
        self.col = col
        self.parent = parent
        self.children = []
        self.root = None
        self.distance_from_start = 0.0
        self.traversed = False
        self.neighbour_direction = neighbour_direction

        if parent is not None:
            self.distance_from_start = parent.distance_from_start + edge_distance
            if reverse_direction:
                self.neighbour_direction = int(
                    neighbour_direction + (-1) ** (neighbour_direction % 2 + 1)
                )
            else:
                self.neighbour_direction = neighbour_direction
        else:
            self.traversed = True
            self.neighbour_direction = neighbour_direction
            self.root = self

    def update_shortest_distance(
        self, parent, edge_distance: float, neighbour_direction: int
    ):
        if self.distance_from_start > edge_distance:
            self.parent = parent
            self.distance_from_start = edge_distance
            if neighbour_direction != 0:
                self.neighbour_direction = int(
                    neighbour_direction + (-1) ** (neighbour_direction % 2 + 1)
                )
            else:
                self.neighbour_direction = 0

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

    def get_updated_neighbours(self, pixel_nodes):
        neighbors_map = problem.neighbors_map
        image_height, image_width = problem.image_shape[:2]
        rgb_matrix = problem.image

        neighbours = []
        for i in range(1, 5):
            neighbour_addition = neighbors_map[i]
            neigh_row, neigh_col = (
                self.row + neighbour_addition[0],
                self.col + neighbour_addition[1],
            )
            neighbour_coords = (neigh_row, neigh_col)

            if not (0 <= neigh_row < image_height and 0 <= neigh_col < image_width):
                continue

            this_color = rgb_matrix[self.row, self.col]
            neighbour_color = rgb_matrix[neigh_row, neigh_col]
            color_dist = color_distance(this_color, neighbour_color)

            if neighbour_coords in pixel_nodes:
                neighbour = pixel_nodes[neighbour_coords]
                if not neighbour.traversed:
                    neighbour.update_shortest_distance(self, color_dist, i)
                    neighbours.append(neighbour)
            else:
                neighbour = PixelNode(neigh_row, neigh_col, self, color_dist, i, True)
                neighbours.append(neighbour)
                pixel_nodes[neighbour_coords] = neighbour

        return neighbours
