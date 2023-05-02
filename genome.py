import numpy as np

from evaluations import (
    calculate_edge_value,
    calculate_segmentation,
    calculate_connectiveness,
    calculate_deviation,
)
import problem


class Genome:
    def __init__(
        self,
        genome: np.ndarray,
        mutation_rate: float,
    ):
        self.genome = genome
        self.mutation_rate = mutation_rate
        self.edge_value = None
        self.connectivity = None
        self.overall_deviation = None
        self.dominated_genomes = []
        self.dominated_by = 0
        self.rank = 0
        self.image_width, self.image_height = problem.image_shape[:2]

    def mutate(self):
        mutation_rate = self.mutation_rate
        while np.random.uniform() < mutation_rate:
            gene = np.random.randint(0, len(self.genome) - 1)
            neigh_dir = np.random.randint(0, 4)
            if gene % self.image_width == 0 and neigh_dir == 2:
                self.genome[gene] = 0
            elif gene % self.image_width == self.image_width - 1 and neigh_dir == 1:
                self.genome[gene] = 0
            elif gene < self.image_width and neigh_dir == 3:
                self.genome[gene] = 0
            elif (
                gene >= self.image_width * self.image_height - self.image_width
                and neigh_dir == 4
            ):
                self.genome[gene] = 0
            else:
                self.genome[gene] = neigh_dir
            mutation_rate -= 1

        # update fitness
        self.update_fitness()

    def update_fitness(self):
        segmentation = calculate_segmentation(self.genome)
        self.edge_value = calculate_edge_value(segmentation)
        self.connectivity = calculate_connectiveness(segmentation)
        self.overall_deviation = calculate_deviation(segmentation)

    def dominate_genome(self, genome):
        self.dominated_genomes.append(genome)
        genome.dominated_by += 1

    def get_next_rank(self, rank):
        if self.dominated_by > 0:
            print(f"Is dominated by {self.dominated_by}")
            return None
        next_rank = []
        self.rank = rank
        for genome in self.dominated_genomes:
            genome.dominated_by -= 1
            if genome.dominated_by == 0:
                next_rank.append(genome)
        self.dominated_genomes = []
        return next_rank
