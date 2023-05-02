import time
import os
import cProfile, pstats, io
from pstats import SortKey


from nsga2 import NSGA2, GAConfig


def main():
    # start profiler
    pr = cProfile.Profile()
    pr.enable()

    before = time.time()
    nsga = NSGA2(size=10)
    ga_config = GAConfig(3, 0.001, 2)
    nsga.run(ga_config)
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
