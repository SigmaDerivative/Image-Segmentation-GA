import numpy as np

import problem
from segmentation import calculate_segmentation


def color_distance(color1, color2):
    """Calculate the distance between two colors.

    Args:
        color1 (np.ndarray): First color.
        color2 (np.ndarray): Second color.

    Returns:
        float: Distance between the two colors.
    """
    return np.sum(np.sqrt((color1 - color2) ** 2))


def calculate_edge_value(segmentation):
    """Calculate the edge value of the genome.

    Args:
        segmentation (np.ndarray): Segmentation of the genome.

    Returns:
        float: Edge value.
    """
    edge_value = 0

    for r in range(problem.image_size[0]):
        for c in range(problem.image_size[1]):
            for neighbor in problem.neighbors_list:
                compare_r = r + neighbor[0]
                compare_c = c + neighbor[1]
                if compare_r < 0 or compare_r >= problem.image_size[0]:
                    continue
                if compare_c < 0 or compare_c >= problem.image_size[1]:
                    continue
                if segmentation[r, c] != segmentation[r + neighbor[0], c + neighbor[1]]:
                    edge_value += color_distance(
                        problem.image[r, c],
                        problem.image[r + neighbor[0], c + neighbor[1]],
                    )
    return edge_value


def calculate_connectiveness(segmentation):
    """Calculate the connectiveness of the genome.

    Args:
        segmentation (np.ndarray): Segmentation of the genome.

    Returns:
        float: Connectiveness.
    """
    connectiveness = 0

    for r in range(problem.image_size[0]):
        for c in range(problem.image_size[1]):
            non_connected_neighbors = 0
            for neighbor in problem.neighbors_list:
                compare_r = r + neighbor[0]
                compare_c = c + neighbor[1]
                if compare_r < 0 or compare_r >= problem.image_size[0]:
                    continue
                if compare_c < 0 or compare_c >= problem.image_size[1]:
                    continue
                if segmentation[r, c] != segmentation[r + neighbor[0], c + neighbor[1]]:
                    non_connected_neighbors += 1
            connectiveness += non_connected_neighbors / len(problem.neighbors_list)
    return connectiveness


def calculate_deviation(segmentation):
    """Calculate the deviation of the genome.

    Args:
        segmentation (np.ndarray): Segmentation of the genome.

    Returns:
        float: Deviation.
    """
    num_segments = np.max(segmentation) + 1

    total_deviation = 0

    for segment in range(num_segments):
        segment_colors = problem.image[segmentation == segment]
        # calculate mean color of segment
        segment_mean_color = np.mean(segment_colors, axis=0)

        # calculate deviation of segment
        squared_deviations = (segment_colors - segment_mean_color) ** 2
        sum_squared_deviations = np.sum(squared_deviations, axis=0)
        segment_deviation = np.sum(np.sqrt(sum_squared_deviations))
        # add deviation to total deviation
        total_deviation += segment_deviation
    return total_deviation


def evaluate_population(population):
    """Evaluate the population.

    Args:
        population (list): List of genomes.

    Returns:
        tuple: Tuple of lists containing the edge values, connectivenesses and deviations of the genomes.
    """
    edge_values = []
    connectivenesses = []
    deviations = []
    for genome in population:
        segmentation = calculate_segmentation(genome)
        edge_value = calculate_edge_value(segmentation)
        connectiveness = calculate_connectiveness(segmentation)
        deviation = calculate_deviation(segmentation)
        edge_values.append(edge_value)
        connectivenesses.append(connectiveness)
        deviations.append(deviation)
    return edge_values, connectivenesses, deviations
