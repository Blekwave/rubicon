from enum import Enum
from collections import namedtuple
import numpy as np
import rubikscube as rc

Record = namedtuple('Record', ('min', 'max', 'mean', 'std'))

def stats_record(entries):
    return Record(min=min(entries), max=max(entries), mean=np.mean(entries),
                  std=np.std(entries))


def run_ga(pop, generations, toolkit):
    """Runs a genetic algorithm.

    Parameters:
    - pop: initial population
    - generations: number of generations the GA should run for
    - toolkit: ga.Toolkit which implements select, best, vary operators
               and a fitness function."""
    fitnesses = [toolkit.fitness(ind) for ind in pop]
    stats = {'fitness': [], 'size': []}

    for gen in range(generations):
        fit_and_pop = list(zip(fitnesses, pop))
        fit_and_offspring = toolkit.select(fit_and_pop)
        best = toolkit.best(fit_and_pop)

        _, offspring = zip(*fit_and_offspring)
        _, best = zip(*best)
        offspring = toolkit.vary(offspring)

        pop = offspring + list(best)
        fitnesses = [toolkit.fitness(ind) for ind in pop]
        fit_stats = stats_record(fitnesses)
        sizes = [len(ind) for ind in pop]
        size_stats = stats_record(sizes)

        log_fmt = "{}\tMin: {}, Avg: {}, Avg size: {}"
        print(log_fmt.format(gen, fit_stats.min, fit_stats.mean, size_stats.mean))
        stats['fitness'].append(fit_stats)
        stats['size'].append(size_stats)

    return best[0], pop, stats

