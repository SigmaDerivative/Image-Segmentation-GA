import time
import os
import shutil
import cProfile, pstats, io
from pstats import SortKey


from nsga2 import NSGA2, GAConfig
from run import evaluate


def main():
    # start profiler
    pr = cProfile.Profile()
    pr.enable()

    # remove previous output
    shutil.rmtree("output/type2/", ignore_errors=True)
    os.makedirs("output/type2/", exist_ok=True)

    before = time.time()
    nsga = NSGA2(size=26)
    ga_config = GAConfig(3, 0.3, 3, 14)
    # nsga.weighted_run(ga_config)
    nsga.run(ga_config)

    # evaluation with PRI
    evaluate(verbose=True)

    after = time.time()
    print("Time taken: ", after - before)

    # collect profile
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    os.makedirs("prof", exist_ok=True)
    ps.dump_stats("prof/main.prof")


if __name__ == "__main__":
    main()
