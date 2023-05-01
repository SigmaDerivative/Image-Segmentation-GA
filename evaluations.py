import numpy as np
import tqdm

import problem
from segmentation import calculate_segmentation


def calculate_edge_value(segmentation):
    """Calculate the edge value of the genome.

    Args:
        segmentation (np.ndarray): Segmentation of the genome.

    Returns:
        float: Edge value.
    """
    edge_value = 0

    for dr, dc in problem.neighbors_list:
        shifted_segmentation = np.roll(segmentation, shift=(dr, dc), axis=(0, 1))
        shifted_image = np.roll(problem.image, shift=(dr, dc), axis=(0, 1))

        mask = segmentation != shifted_segmentation
        color_dists = np.sqrt(np.sum((problem.image - shifted_image) ** 2, axis=-1))

        edge_value += np.sum(color_dists * mask)

    return edge_value


def calculate_connectiveness(segmentation):
    """Calculate the connectiveness of the genome.

    Args:
        segmentation (np.ndarray): Segmentation of the genome.

    Returns:
        float: Connectiveness.
    """
    connectiveness = 0
    neighbors = np.array(problem.neighbors_list)

    for neighbor in neighbors:
        shifted_segmentation = np.roll(segmentation, shift=tuple(neighbor), axis=(0, 1))

        # Handle border cases by masking out invalid comparisons
        mask = np.ones_like(segmentation, dtype=bool)
        if neighbor[0] > 0:
            mask[: neighbor[0], :] = False
        elif neighbor[0] < 0:
            mask[neighbor[0] :, :] = False

        if neighbor[1] > 0:
            mask[:, : neighbor[1]] = False
        elif neighbor[1] < 0:
            mask[:, neighbor[1] :] = False

        non_connected_neighbors = np.sum((segmentation != shifted_segmentation) & mask)
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
        segment_mask = segmentation == segment
        segment_colors = problem.image[segment_mask]

        segment_mean_color = np.mean(segment_colors, axis=0)
        squared_deviations = np.square(segment_colors - segment_mean_color)
        sum_squared_deviations = np.sum(squared_deviations, axis=0)
        segment_deviation = np.sqrt(np.sum(sum_squared_deviations))

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
    for genome in tqdm.tqdm(population):
        segmentation = calculate_segmentation(genome)
        edge_values.append(calculate_edge_value(segmentation))
        connectivenesses.append(calculate_connectiveness(segmentation))
        deviations.append(calculate_deviation(segmentation))
    return edge_values, connectivenesses, deviations


def evaluate_population_from_segments(segments):
    """Evaluate the population from their segments.

    Args:
        segments (list): List of segments.

    Returns:
        tuple: Tuple of lists containing the edge values, connectivenesses and deviations of the genomes.
    """
    edge_values = []
    connectivenesses = []
    deviations = []
    for segmentation in tqdm.tqdm(segments):
        segmentation = segmentation.reshape(
            (problem.image_shape[0], problem.image_shape[1])
        ).astype(int)
        edge_values.append(calculate_edge_value(segmentation))
        connectivenesses.append(calculate_connectiveness(segmentation))
        deviations.append(calculate_deviation(segmentation))
    return edge_values, connectivenesses, deviations


def main():
    """Main function."""
    from visualization import plot_type_2

    population = np.random.randint(
        1, 3, (4, problem.image_shape[0], problem.image_shape[1])
    )
    # population = generate_random_population(3)
    evals = evaluate_population(population)
    # population = np.random.randint(
    #     0, 2, (10, problem.image_size[0] * problem.image_size[1])
    # )
    # print(population.shape)
    # evals = evaluate_population_from_segments(population)
    print(evals)
    segment = calculate_segmentation(population[0])
    plot_type_2(segment)


if __name__ == "__main__":
    import os
    import cProfile, pstats, io
    from pstats import SortKey

    # start profiler
    pr = cProfile.Profile()
    pr.enable()

    main()
    # collect profile
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    os.makedirs("prof", exist_ok=True)
    ps.dump_stats("prof/eval.prof")
