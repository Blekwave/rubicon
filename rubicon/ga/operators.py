import random
import heapq


def sel_tourn(fit_and_pop, num_offspring, k):
    """Select offspring from a population via tournament selection.

    Parameters:
    - fit_and_pop: list of fitness-individual tuples
    - num_offspring: number of offspring to be generated
    - k: tournament size

    Returns a list of selected offspring fitness-individual pairs."""
    offspring = []
    for _ in range(num_offspring):
        tournament = random.sample(fit_and_pop, k)
        offspring.append(min(tournament))
    return offspring


def sel_best(fit_and_pop, num_offspring):
    """Select the n best offspring from a population.

    Parameters:
    - fit_and_pop: list of fitness-individual tuples
    - num_offspring: number of offspring to be generated

    Returns a list of selected offspring fitness-individual pairs."""
    return heapq.nsmallest(num_offspring, fit_and_pop)


def vary(pop, toolkit, cx_prob, mut_prob):
    """Vary population via crossover and mutation, possibly both.

    Note that crossover and mutation probabilities are independent.

    Parameters:
    - pop: list of individuals
    - toolkit: object containing mate and mutate operators
    - cx_prob: crossover probability
    - mut_prob: mutation probability

    Returns the population after variation."""
    half = len(pop) // 2

    # crossover
    pop_after_cx = []
    for a, b in zip(pop[:half], pop[half:]):
        roll = random.random()
        if roll < cx_prob:
            a, b = toolkit.mate(a, b)
        pop_after_cx.append(a)
        pop_after_cx.append(b)

    # if there's an odd number of individuals in the population, the
    # last one won't have been copied to the new list
    if len(pop) % 2:
        pop_after_cx.append(pop[-1])

    # mutation
    pop_after_mut = []
    for ind in pop_after_cx:
        roll = random.random()
        if roll < mut_prob:
            ind, = toolkit.mutate(ind)
        pop_after_mut.append(ind)

    return pop_after_mut


def size_limit(operator, limit):
    """Limit a individual mutation/mating operator's output size.

    If any of the individuals returned by the operator violates the
    limit, the input individuals are returned instead.

    Parameters:
    - operator: mutation/mating operator, which takes as input N
                individuals and returns N new individuals. It is
                paramount that this function does not change the
                input individuals or take any other parameters
                (use partial functions!)
    - limit: maximum individual size

    Returns the decorated operator."""
    def limited_function(*input_inds):
        output_inds = operator(*input_inds)
        for ind in output_inds:
            if len(ind) > limit:
                return input_inds
        return output_inds
    return limited_function

