import math
import json
import datetime
import time
import os
import sys
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


def single_run(toolkit, run_dir, verbose=True):
    config = toolkit.config

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    start_time = time.time()

    pop = toolkit.init_pop()
    fit_and_pop, stats = run_ga(pop, config['GA']['Gens'], toolkit, verbose)

    best_fitness, best = min(fit_and_pop)

    end_time = time.time()
    duration = end_time - start_time  # in seconds

    best_final_cube = rc.apply_moves(toolkit.initial_cube, best)
    if verbose:
        pprint.pprint(best, indent=4, compact=True)
        print("Fitness:", best_fitness)
        rc.print_3d_cube(best_final_cube)

    log_run(run_dir, config, stats, duration)
    log_individuals(run_dir, fit_and_pop, best_final_cube)

    return (best_fitness, best), pop, stats


def multi_run(toolkit, all_runs_dir):
    config = toolkit.config

    fit_and_best = []
    run_stats = []

    start_time = time.time()

    for run in range(config['Runs']):
        digits = int(math.log(config['Runs'], 10)) + 1
        run_id = str(run).zfill(digits)
        run_dir = os.path.join(all_runs_dir, "run_{}".format(run_id))
        run_fit_and_best, _, stats = single_run(toolkit, run_dir, verbose=True)
        log_fmt = "Run {}: Fitness {}\nBest: {}"
        print(log_fmt.format(run, *run_fit_and_best))
        fit_and_best.append(run_fit_and_best)
        run_stats.append(stats)

    fitness, best = min(fit_and_best)
    print('Best:', best)
    print('Fitness:', fitness)
    best_final_cube = rc.apply_moves(toolkit.initial_cube, best)
    rc.print_3d_cube(best_final_cube)

    end_time = time.time()
    duration = end_time - start_time  # in seconds

    summary = summarize_stats(run_stats)

    log_multi_run(all_runs_dir, config, summary, duration)
    log_individuals(all_runs_dir, fit_and_best, best_final_cube)


def main(pool=None):
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)

    run_name = config['Name']

    toolkit = RubiconToolkit(config)
    if pool:
        toolkit.map = pool.map

    timestr = datetime.datetime.now().strftime("%Y%m%d-%Hh%Mm%Ss")
    all_runs_dir = os.path.join(RUNS_DIR, "{}-{}".format(timestr, run_name))

    print("Start of execution:", timestr)
    print("{} runs".format(config['Runs']))

    if config['Runs'] == 1:
        single_run(toolkit, all_runs_dir)
    elif config['Runs'] > 1:
        multi_run(toolkit, all_runs_dir)


if __name__ == '__main__':
    main(pool=mp.Pool())
