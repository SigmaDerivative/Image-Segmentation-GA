"""Implementation of NSGA-II algorithm."""
from dataclasses import dataclass
import random
from typing import List
import time

import numpy as np
from tqdm import tqdm

import problem
from genome import Genome
from initializations import generate_mst_genome, generate_clustered_genome
from trees import get_crossover_indices, get_crossover_indices_from_genomes
from visualization import plot_type_3, plot_type_2, plot_type_1
from segmentation import calculate_segmentation


@dataclass
class GAConfig:
    num_epochs: int
    mutation_rate: float
    crossover_sections: int
    num_children: int


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
            if np.random.uniform() < 0.5:
                genome = generate_mst_genome(flat=True)
            else:
                genome = generate_clustered_genome(flat=True)
            g = Genome(genome, self.mutation_rate)
            g.update_fitness()
            self.current_population.append(g)

    def finish(self, no_rank=False) -> None:
        i = 0
        if no_rank:
            genomes = self.current_population
        else:
            ranks = self.get_ranks()
            genomes = ranks[0]
        for genome in genomes:
            seg = calculate_segmentation(genome.genome)
            plot_type_2(seg)
            plot_type_1(seg)
            print(f"Solution {i}:")
            print(f"Edge value: {genome.edge_value}")
            print(f"Connectivity: {genome.connectivity}")
            print(f"Deviation: {genome.overall_deviation}")
            print()
            i += 1
            if i == 5:
                break

    def run(self, config: GAConfig) -> None:
        self.initiate(config)

        for _ in range(config.num_epochs):
            self.epoch(config.num_children)

        self.finish()

    def epoch(self, num_children: int) -> None:
        print("Starting generation number " + str(self.generation))
        before = time.time()

        while len(self.next_population) < self.size + num_children:
            tournament_winners = self.get_tournament_winners()
            self.crossover(
                tournament_winners[0], tournament_winners[1], self.crossover_sections
            )

        current_population = []

        ranks = self.get_ranks()

        for rank in enumerate(ranks):
            if len(rank) + len(current_population) <= self.size:
                current_population += rank[1]
            elif len(current_population) < self.size:
                random.shuffle(rank[1])
                index = 0
                while len(current_population) < self.size:
                    current_population.append(rank[1][index])
                    index += 1
            else:
                break

        self.next_population = current_population.copy()

        self.generation += 1

        after = time.time()
        print("Generation took " + str(after - before) + " seconds.")

    def weighted_run(self, config: GAConfig) -> None:
        self.initiate(config)

        for generation in range(config.num_epochs):
            print("Starting generation number " + str(generation))
            before = time.time()
            while len(self.next_population) < self.size:
                tournament_winners = self.get_weighted_tournament_winners()
                self.crossover(
                    tournament_winners[0],
                    tournament_winners[1],
                    self.crossover_sections,
                )
            self.current_population = list(self.next_population)
            self.next_population = []
            self.generation += 1
            after = time.time()
            print("Generation took " + str(after - before) + " seconds.")

        self.finish(no_rank=True)

    def get_tournament_winners(self):
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
        winners = [None, None]
        max_fitness = [float("-inf"), float("-inf")]

        fitnesses = [[0.0] for i in range(3)]
        for genome in self.current_population:
            fitnesses[0].append(genome.edge_value)
            fitnesses[1].append(genome.connectivity)
            fitnesses[2].append(genome.overall_deviation)
        fitnesses[0].sort()
        fitnesses[1].sort()
        fitnesses[2].sort()

        for genome in self.current_population:
            genome_fitness = 0.0
            genome_fitness += (-genome.edge_value) * (
                fitnesses[0][-1] - fitnesses[0][0]
            )
            genome_fitness += (-genome.connectivity + fitnesses[1][-1]) * (
                fitnesses[1][-1] - fitnesses[1][0]
            )
            genome_fitness += (-genome.overall_deviation + fitnesses[2][-1]) * (
                fitnesses[2][-1] - fitnesses[2][0]
            )

            if genome_fitness > max_fitness[0]:
                max_fitness[1] = max_fitness[0]
                winners[1] = winners[0]
                max_fitness[0] = genome_fitness
                winners[0] = genome
            elif genome_fitness > max_fitness[1]:
                max_fitness[1] = genome_fitness
                winners[1] = genome

        return winners

    def crossover(self, genome1, genome2, number_of_selections):
        raw_genome_1 = genome1.genome[:]
        raw_genome_2 = genome2.genome[:]
        if np.random.uniform() > 0.5:
            crossover_indices = get_crossover_indices(number_of_selections)
        else:
            crossover_indices = get_crossover_indices_from_genomes(
                [raw_genome_1, raw_genome_2]
            )
        for i in range(len(crossover_indices)):
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

    def get_ranks(self) -> List[List[Genome]]:
        def dominates(genome1, genome2):
            less_equal = (
                genome1.edge_value <= genome2.edge_value
                and genome1.connectivity <= genome2.connectivity
                and genome1.overall_deviation <= genome2.overall_deviation
            )
            strictly_less = (
                genome1.edge_value < genome2.edge_value
                or genome1.connectivity < genome2.connectivity
                or genome1.overall_deviation < genome2.overall_deviation
            )
            return less_equal and strictly_less

        for genome in self.next_population:
            for dominated_genome in self.next_population:
                if genome != dominated_genome and dominates(genome, dominated_genome):
                    genome.dominate_genome(dominated_genome)

        rank0 = [genome for genome in self.next_population if genome.dominated_by == 0]
        rank1 = [genome for genome in rank0 if genome.connectivity < 100.0]

        if rank0 and rank1:
            for genome in rank1:
                rank0[0].dominate_genome(genome)
                rank0.remove(genome)

        ranked_nodes = [rank0]
        rank = 0

        while rank < len(ranked_nodes):
            rank_n = []
            for genome in ranked_nodes[rank]:
                next_rank = genome.get_next_rank(rank)
                if next_rank:
                    rank_n.extend(next_rank)
            if rank_n:
                ranked_nodes.append(rank_n)
            rank += 1

        return ranked_nodes
