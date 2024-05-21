from utils import *
from hyperparams import *

N = 1
if __name__ == "__main__":
    while N > 0:
        N -= 1
        args_dict = sample_dict(experiment_space, 1)
        cached_run_experiments = cache(run_experiment)
        cached_run_experiments(args_dict)
        print(f"N = {N}")
