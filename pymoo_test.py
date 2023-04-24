import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling, IntegerRandomSampling
from pymoo.optimize import minimize

from initializations import (
    generate_k_meaned_segmentation_flat,
    generate_random_genome_from_MST,
)
from evaluations import evaluate_population, evaluate_population_from_segments
from segmentation import calculate_segmentation
from visualization import plot_type_2
import problem


class MyProblem(Problem):
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.reshape(x, (-1, problem.image_size[0], problem.image_size[1]))
        f1, f2, f3 = evaluate_population(x)
        # f1, f2, f3 = evaluate_population_from_segments(x)
        n_f1 = np.negative(f1)
        evaluation = np.column_stack([n_f1, f2, f3])

        out["F"] = evaluation


def main():
    n_vars = problem.image_size[0] * problem.image_size[1]
    pymoo_problem = MyProblem(n_var=n_vars, n_obj=3, xl=1, xu=20, vtype=int)
    algorithm = NSGA2(
        pop_size=100,
        sampling=generate_random_genome_from_MST(),
        n_offsprings=40,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    stop_criteria = ("n_gen", 2)
    results = minimize(
        problem=pymoo_problem,
        algorithm=algorithm,
        termination=stop_criteria,
        verbose=True,
    )
    print(results.X)
    print(results.F)
    for _, res_x in enumerate(results.X):
        genome = res_x.reshape(problem.image_size[0], problem.image_size[1]).astype(int)
        segmentation = calculate_segmentation(genome)
        # visualize the segments
        # plot_type_2(genome)
        plot_type_2(segmentation)


if __name__ == "__main__":
    main()
