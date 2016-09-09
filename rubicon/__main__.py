import datetime
import time
import os
from collections import namedtuple

import rubikscube as rc
from ga import run_ga
from rubicon_toolkit import RubiconToolkit
from log_tools import log_run, log_individuals

THIS_FILE = os.path.realpath(__file__)
RUNS_DIR = os.path.join(os.path.dirname(THIS_FILE), "../runs")


def make_run_dir(timestr, label):
    dir_name = os.path.join(RUNS_DIR, "{}-runs".format(timestr))
    if label:
        dir_name = os.path.join(dir_name, label)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

config = {
    'GA': {
        'InitMinSize': 5,
        'InitMaxSize': 15,
        'MutMinSize': 1,
        'MutMaxSize': 10,
        'IndMaxSize': 100,
        'PopSize': 200,
        'Gens': 500,
        'TournSize': 3,
        'NumElitism': 1,
        'CxProb': 0.8,
        'MutProb': 0.35,
    },
    'Rubiks': {
        'InitialPath': 'inputs/in1/in1'
    }
}

Timestamps = namedtuple("Timestamps", ["start", "duration"])


def main(run_label=None):
    """Runs an instance of the genetic algorithm for solving a cube.

    Parameters:
    - run: labels this specific run. Should be used when running many
           instances in parallel, in order to avoid folder name
           conflict."""

    toolkit = RubiconToolkit(config)

    start_timestr = datetime.datetime.now().strftime("%Y%m%d-%Hh%Mm%Ss")
    start_time = time.time()

    pop = toolkit.init_pop()
    best, pop, stats = run_ga(pop, config['GA']['Gens'], toolkit)

    end_time = time.time()
    duration = end_time - start_time  # in seconds

    print(best)
    best_final_cube = rc.apply_moves(toolkit.initial_cube, best)
    rc.print_3d_cube(best_final_cube)

    run_dir = make_run_dir(start_timestr, run_label)
    timestamps = Timestamps(start=start_timestr, duration=duration)

    log_run(run_dir, config, stats, timestamps)
    log_individuals(run_dir, best, pop, best_final_cube)


if __name__ == '__main__':
    main()
