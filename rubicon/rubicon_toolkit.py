import random
import math
from functools import partial

import rubikscube as rc
import ga.operators as ops

from ga import Toolkit
from graph_fitness import solution_distance
from cube_fitness import wrong_color_facelets, wrong_cubelets


def combined_fitness(ind, initial_cube):
    """Evaluate an individual based on an initial cube

    Combines four different fitness functions:
    - The count of wrong cubelets (orientation doesn't matter)
    - The count of facelets with the wrong color
    - The graph solution distance of the evaluated cube
    - The size of the individual

    Applies different, hard-coded coefficients to each function.

    Returns a fitness value."""
    cube = rc.apply_moves(initial_cube, ind)
    return (wrong_cubelets(cube) +
            wrong_color_facelets(cube) / 2.4 +
            solution_distance(cube) / 4.8 +
            math.log(len(ind)) / 30)


def create_ind(min_size, max_size):
    """Randomly create an individual.

    In this GA, an individual is an array of integer indiced which
    correspond to cube moves.

    Parameters:
    - min_size: minimum size of the individual
    - max_size: maximum size of the individual"""
    return [random.randint(0, len(rc.moves) - 1)
            for _ in range(random.randint(min_size, max_size))]


def cx_point(a, b):
    """Perform single-point crossover between two individuals.

    Parameters:
    - a, b: individuals to be crossed over

    Returns the crossed-over individuals."""
    # crossover points
    i = random.randint(0, len(a) - 1)
    j = random.randint(0, len(b) - 1)
    return a[:i] + b[j:], b[:j] + a[i:]


def mutate_replace(ind, min_size, max_size):
    """Mutate an individual by replacing a fragment with a new one.

    Parameters:
    - ind: individual to be mutated
    - min_size: minimum size of the new/removed fragment
    - max_size: minimum size of the new/removed fragment

    Returns the mutated individual."""

    removed_fragment_size = random.randint(min_size, min(max_size, len(ind)))
    remove_begin = random.randint(0, len(ind) - removed_fragment_size)
    remove_end = remove_begin + removed_fragment_size

    new_fragment = create_ind(min_size, max_size)
    return (ind[:remove_begin] + new_fragment + ind[remove_end:],)


class RubiconToolkit(Toolkit):
    """Toolkit for the Rubik's Cube GA solver.

    Utilizes relevant operators for the problem:
    - create: random array of movements
    - select: tournament with elitism
    - vary: crossover, mutation and reproduction on independent
            probabilities
    - mate: single-point crossover
    - mutate: random fragment replacement
    - fitness: combined fitness described in combined_fitness's
               docstring."""
    def __init__(self, config):
        """Initialize the toolkit, binding the configuration to the
        operators.

        Parameters:
        - config: configuration object with the execution parameters."""
        self.config = config
        c = config['GA']

        # create individual
        create = partial(create_ind, min_size=c['InitMinSize'],
                         max_size=c['InitMaxSize'])
        self.create = create

        # select offspring
        not_elitist = c['PopSize'] - c['NumElitism']
        select = partial(ops.sel_tourn, num_offspring=not_elitist,
                         k=c['TournSize'])
        self.select = select

        # select best (elitism)
        best = partial(ops.sel_best, num_offspring=c['NumElitism'])
        self.best = best

        # vary offspring
        vary = partial(ops.vary, toolkit=self, cx_prob=c['CxProb'],
                       mut_prob=c['MutProb'])
        self.vary = vary

        # mate two individuals
        mate = ops.size_limit(cx_point, c['IndMaxSize'])
        self.mate = mate

        # mutate an individual
        mutate = partial(mutate_replace, min_size=c['MutMinSize'],
                         max_size=c['MutMaxSize'])
        mutate = ops.size_limit(mutate, c['IndMaxSize'])
        self.mutate = mutate

        # compute fitness of an individual
        initial_cube = rc.from_file(config['Rubiks']['InitialPath'],
                                    flatten=True)
        self.initial_cube = initial_cube
        fitness = partial(combined_fitness, initial_cube=initial_cube)
        self.fitness = fitness

    def init_pop(self):
        """Initialize a new population of Rubik's Cube GA individuals.

        Returns a list of individuals."""
        return super().init_pop(self.config['GA']['PopSize'])
