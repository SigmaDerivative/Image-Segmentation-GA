"""Implementation of NSGA-II algorithm."""
from dataclasses import dataclass

import numpy as np

from initializations import generate_random_population
from evaluations import evaluate_population


@dataclass
class EpochConfig:
    num_parents: int
    num_new_random_individuals: int
    n_children: int
    mate_elite_prob_factor: float = 2.0
    mutation_m: int = 4


class NSGA2:
    def __init__(self, size: int) -> None:
        genomes = generate_random_population(population_size=size)
        edge_values, connectivities, deviations = evaluate_population(genomes)
        self.genomes = genomes
        self.edge_values = edge_values
        self.connectivities = connectivities
        self.deviations = deviations
        self.size = size
        self.epoch_number = 0

    def __add__(self, other: "NSGA2") -> "NSGA2":
        genomes = np.concatenate((self.genomes, other.genomes))
        edge_values = np.concatenate((self.edge_values, other.edge_values))
        connectivities = np.concatenate((self.connectivities, other.connectivities))
        deviations = np.concatenate((self.deviations, other.deviations))
        ga = NSGA2(size=0)
        ga.genomes = genomes
        ga.edge_values = edge_values
        ga.connectivities = connectivities
        ga.deviations = deviations
        ga.size = self.size + other.size
        ga.epoch_number = self.epoch_number + other.epoch_number
        return ga

    def sort_population_(self) -> None:
        pass
        # TODO

    def epoch(
        self,
        config: EpochConfig,
    ) -> None:
        pass
        # TODO
        # get parent candidates
        # parent_genomes, _, _ = elitist(
        #     genomes=self.genomes,
        #     fitness=self.fitness,
        #     valids=self.valids,
        #     num_elites=config.num_parents,
        # )

        # # random repair function
        # if np.random.uniform() < 0.4:
        #     repair_func = repair_random
        # elif np.random.uniform() < 0.4:
        #     repair_func = repair_greedy
        # else:
        #     repair_func = repair_random_uniform

        # # crossover
        # if np.random.uniform() < 0.4:
        #     child_genomes = deterministic_isolated_mating(
        #         parent_genomes, config.n_destroys, repair_func
        #     )
        # else:
        #     child_genomes = stochastic_elitist_mating(
        #         parent_genomes,
        #         config.n_destroys,
        #         repair_func,
        #         config.n_children,
        #         config.mate_elite_prob_factor,
        #     )
        # mutated_genomes = mutate_population(population=child_genomes, m=5)

        # # create new individuals
        # if config.num_new_clustered_individuals > 0:
        #     new_genomes_cluster = generate_cluster_population(
        #         size=config.num_new_clustered_individuals
        #     )
        #     new_genomes_cluster = mutate_population(
        #         population=new_genomes_cluster, m=config.mutation_m
        #     )
        # new_genomes_random = generate_random_population(
        #     size=config.num_new_random_individuals
        # )
        # new_genomes_random = mutate_population(
        #     population=new_genomes_random, m=config.mutation_m
        # )

        # # combine all genomes
        # if config.num_new_clustered_individuals > 0:
        #     total_genomes = np.vstack(
        #         (
        #             parent_genomes,
        #             new_genomes_cluster,
        #             new_genomes_random,
        #             mutated_genomes,
        #         )
        #     )
        # else:
        #     total_genomes = np.vstack(
        #         (parent_genomes, new_genomes_random, mutated_genomes)
        #     )
        # total_fitness, total_valids = evaluate_population(
        #     total_genomes, config.penalize_invalid
        # )
        # # survival of the fittest
        # surviver_genomes, surviver_fitness, surviver_valids = elitist(
        #     genomes=total_genomes,
        #     fitness=total_fitness,
        #     valids=total_valids,
        #     num_elites=self.size,
        # )

        # # update population
        # self.genomes = surviver_genomes
        # self.fitness = surviver_fitness
        # self.valids = surviver_valids

        # # update epoch number
        # self.epoch_number += 1


def main():
    pop = NSGA2(size=10)
    print(pop.genomes)


if __name__ == "__main__":
    main()
