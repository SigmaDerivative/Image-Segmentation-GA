"""Implementation of NSGA-II algorithm."""
from dataclasses import dataclass
import random

import numpy as np
from tqdm import tqdm

from genome import Genome
from initializations import generate_mst_genome
from trees import get_crossover_indices
import problem
from visualization import plot_type_3, plot_type_2, plot_type_1
from segmentation import calculate_segmentation


@dataclass
class GAConfig:
    num_epochs: int
    mutation_rate: float
    crossover_sections: int


class NSGA2:
    def __init__(self, size: int) -> None:
        self.genomes = []
        self.size = size
        self.epoch_number = 0
        self.generation = 0
        self.current_population = []
        self.next_population = []

        self.height, self.width = problem.image_shape[:2]
        self.crossover_sections = None
        self.mutation_rate = None

    def initiate(self, config: GAConfig) -> None:
        print("Initiating population...")
        self.mutation_rate = config.mutation_rate
        self.crossover_sections = config.crossover_sections
        for _ in tqdm(range(self.size)):
            genome = generate_mst_genome(flat=True)
            self.current_population.append(Genome(genome, self.mutation_rate))

    def run(self, config: GAConfig) -> None:
        self.initiate(config)

        for _ in range(config.num_epochs):
            self.epoch()

        for genome in self.current_population:
            seg = calculate_segmentation(genome.genome)
            plot_type_2(seg)
            plot_type_1(seg)
        print("Done!")

    def epoch(self) -> None:
        print("Starting generation nr " + str(self.generation))

        # Keep adding new genomes to the next population until it reaches the pool size multiplied by 2
        while len(self.next_population) < self.size * 2:
            tournament_winners = self.get_tournament_winners()
            self.crossover(
                tournament_winners[0], tournament_winners[1], self.crossover_sections
            )

        # Create a new list for the current population
        current_population = []

        # Get the ranks for the genomes
        ranks = self.get_ranks()

        # Loop through the ranks
        for rank in enumerate(ranks):
            # If adding the entire rank to the current population won't exceed the pool size
            if len(rank) + len(current_population) <= self.size:
                current_population += rank[1]
            # If adding the entire rank will exceed the pool size,
            # shuffle the rank and add as many genomes as needed to reach the pool size
            elif len(current_population) < self.size:
                random.shuffle(rank[1])
                index = 0
                while len(current_population) < self.size:
                    current_population.append(rank[1][index])
                    index += 1
            # If the current population already has the desired pool size, exit the loop
            else:
                break

        # Copy the current population to the next population
        self.next_population = current_population.copy()

        # Increment the generation count
        self.generation += 1

    def weighted_run(self, config: GAConfig) -> None:
        self.initiate(config)

        for generation in range(config.num_epochs):
            print("Starting generation nr " + str(generation))
            while len(self.next_population) < self.size:
                tournament_winners = self.get_weighted_tournament_winners()
                self.crossover(
                    tournament_winners[0],
                    tournament_winners[1],
                    self.crossover_sections,
                )
            self.current_population = list(self.next_population)
            self.next_population = []

        # Draws image
        for genome in self.current_population:
            seg = calculate_segmentation(genome.genome)
            plot_type_2(seg)
            plot_type_1(seg)

    def mutate_genome(self, genome):
        genome.mutate()

    def get_tournament_winners(self):
        # current_population. Ranks should be up to date.
        winners = []
        for tournaments in range(2):
            participant_indices = np.random.randint(
                0, len(self.current_population) - 1, 2
            )
            participant1 = self.current_population[int(participant_indices[0])]
            participant2 = self.current_population[int(participant_indices[1])]
            if participant1.rank < participant2.rank:
                winners.append(participant1)
            else:
                winners.append(participant2)
        return winners

    def get_weighted_tournament_winners(self):
        winners = []
        fitnesses = [[0.0] for i in range(3)]
        for genome in self.current_population:
            fitnesses[0].append(genome.edge_value)
            fitnesses[1].append(genome.connectivity)
            fitnesses[2].append(genome.overall_deviation)
        fitnesses[0].sort()
        fitnesses[1].sort()
        fitnesses[2].sort()
        total_fitness = 0.0
        for genome in self.current_population:
            total_fitness += (-genome.edge_value) * (fitnesses[0][-1] - fitnesses[0][0])
            total_fitness += (-genome.connectivity + fitnesses[1][-1]) * (
                fitnesses[1][-1] - fitnesses[1][0]
            )
            total_fitness += (-genome.overall_deviation + fitnesses[2][-1]) * (
                fitnesses[2][-1] - fitnesses[2][0]
            )
        for tournaments in range(2):
            genome_fit = np.random.random() * total_fitness
            running_fit = 0.0
            for genome in self.current_population:
                running_fit += (-genome.edge_value) * (
                    fitnesses[0][-1] - fitnesses[0][0]
                )
                running_fit += (-genome.connectivity + fitnesses[1][-1]) * (
                    fitnesses[1][-1] - fitnesses[1][0]
                )
                running_fit += (-genome.overall_deviation + fitnesses[2][-1]) * (
                    fitnesses[2][-1] - fitnesses[2][0]
                )
                if running_fit >= genome_fit:
                    winners.append(genome)
                    break
        return winners

    def crossover(self, genome1, genome2, number_of_selections):
        crossover_indices = get_crossover_indices(number_of_selections)
        raw_genome_1 = genome1.genome[:]
        raw_genome_2 = genome2.genome[:]
        for i in range(0, number_of_selections, 2):
            indexes = crossover_indices[i]
            for index in indexes:
                raw_genome_1[index], raw_genome_2[index] = (
                    raw_genome_2[index],
                    raw_genome_1[index],
                )
        child1 = Genome(raw_genome_1, self.mutation_rate)
        child1.mutate()
        self.next_population.append(child1)
        child2 = Genome(raw_genome_2, self.mutation_rate)
        child2.mutate()
        self.next_population.append(child2)

    def get_crowding_list(self, rank):
        crowding_list = list(rank)
        objectives = [[], [], []]
        for genome in rank:
            objectives[0].append(genome.edge_value)
            objectives[1].append(genome.connectivity)
            objectives[2].append(genome.overall_deviation)
        sorted(objectives[0])
        sorted(objectives[1])
        sorted(objectives[2])
        sorted_genomes = list(rank)
        for index in range(3):
            new_sorted_genomes = []
            for genome in sorted_genomes:
                if genome.edge_value == objectives[0][0]:
                    new_sorted_genomes.append(genome)
                    objectives[0].pop(0)
                sorted_genomes = list(new_sorted_genomes)
        return crowding_list

    def get_ranks(self):
        for genome in self.next_population:
            for dominated_genome in self.next_population:
                if genome != dominated_genome:
                    if (
                        genome.edge_value <= dominated_genome.edge_value
                        and genome.connectivity <= dominated_genome.connectivity
                        and genome.overall_deviation
                        <= dominated_genome.overall_deviation
                    ):
                        if (
                            genome.edge_value < dominated_genome.edge_value
                            or genome.connectivity < dominated_genome.connectivity
                            or genome.overall_deviation
                            < dominated_genome.overall_deviation
                        ):
                            genome.dominate_genome(dominated_genome)
        ranked_nodes = []
        rank0 = []
        rank1 = []
        for genome in self.next_population:
            if genome.dominated_by == 0:
                if genome.connectivity < 100.0:
                    rank1.append(genome)
                rank0.append(genome)
        if len(rank0) > 0 and len(rank1) > 0:
            for genome in rank1:
                rank0[0].dominate_genome(genome)
                rank0.remove(genome)
        ranked_nodes.append(rank0)
        rank = 0
        while rank < len(ranked_nodes):
            rank_n = []
            for genome in ranked_nodes[rank]:
                next_rank = genome.get_next_rank(rank)
                if len(next_rank) > 0:
                    rank_n.extend(next_rank)
            if len(rank_n) > 0:
                ranked_nodes.append(rank_n)
            rank += 1
        return ranked_nodes


def main():
    import time

    before = time.time()
    nsga = NSGA2(size=25)
    ga_config = GAConfig(10, 0.001, 2)
    nsga.run(ga_config)
    after = time.time()
    print("Time taken: ", after - before)


if __name__ == "__main__":
    main()
