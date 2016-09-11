import math
import datetime
import time
import os
import pprint
import multiprocessing as mp
from collections import namedtuple

import numpy as np

import rubikscube as rc
from ga import run_ga, summarize_stats
from rubicon_toolkit import RubiconToolkit
from log_tools import log_run, log_multi_run, log_individuals

THIS_FILE = os.path.realpath(__file__)
RUNS_DIR = os.path.join(os.path.dirname(THIS_FILE), "../runs")

CONFIG = {
    'GA': {
        'InitMinSize': 5,
        'InitMaxSize': 15,
        'MutMinSize': 1,
        'MutMaxSize': 10,
        'IndMaxSize': 100,
        'PopSize': 300,
        'Gens': 100,
        'TournSize': 3,
        'NumElitism': 1,
        'CxProb': 0.9,
        'MutProb': 0.1,
    },
    'Rubiks': {
        'InitialPath': 'inputs/in1/in1'
    },
    'Runs': 30
}


def single_run(toolkit, run_dir, verbose=True):
    config = toolkit.config

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    start_time = time.time()

    pop = toolkit.init_pop()
    best, pop, stats = run_ga(pop, config['GA']['Gens'], toolkit, verbose)
    best_fitness = toolkit.fitness(best)

    end_time = time.time()
    duration = end_time - start_time  # in seconds

    best_final_cube = rc.apply_moves(toolkit.initial_cube, best)
    if verbose:
        pprint.pprint(best, indent=4, compact=True)
        print("Fitness:", best_fitness)
        rc.print_3d_cube(best_final_cube)

    log_run(run_dir, config, stats, duration)
    log_individuals(run_dir, best, pop, best_final_cube)

    return (best_fitness, best), pop, stats


def multi_run(toolkit, all_runs_dir):
    config = toolkit.config

    fitness_and_best = []
    run_stats = []

    start_time = time.time()

    for run in range(config['Runs']):
        digits = int(math.log(config['Runs'], 10)) + 1
        run_id = str(run).zfill(digits)
        run_dir = os.path.join(all_runs_dir, "run_{}".format(run_id))
        run_fitness_and_best, _, stats = single_run(toolkit, run_dir, verbose=True)
        log_fmt = "Run {}: Fitness {}\nBest: {}"
        print(log_fmt.format(run, *run_fitness_and_best))
        fitness_and_best.append(run_fitness_and_best)
        run_stats.append(stats)

    fitness, best = min(fitness_and_best)
    print('Best:', best)
    print('Fitness:', fitness)
    best_final_cube = rc.apply_moves(toolkit.initial_cube, best)
    rc.print_3d_cube(best_final_cube)

    end_time = time.time()
    duration = end_time - start_time  # in seconds

    summary = summarize_stats(run_stats)
    _, best_inds = zip(*fitness_and_best)

    log_multi_run(all_runs_dir, config, summary, duration)
    log_individuals(all_runs_dir, best, best_inds, best_final_cube)


def main(pool=None):
    config = CONFIG
    toolkit = RubiconToolkit(config)
    if pool:
        toolkit.map = pool.map

    timestr = datetime.datetime.now().strftime("%Y%m%d-%Hh%Mm%Ss")
    all_runs_dir = os.path.join(RUNS_DIR, "{}-runs".format(timestr))

    print("Start of execution:", timestr)
    print("{} runs".format(config['Runs']))

    if config['Runs'] == 1:
        single_run(toolkit, all_runs_dir)
    elif config['Runs'] > 1:
        multi_run(toolkit, all_runs_dir)


if __name__ == '__main__':
    main(pool=mp.Pool())
