class Toolkit:
    """Container for the operators and the fitness function of a GA
    run. Must be specialized before being used."""

    def create(self):
        """Create a single individual."""
        raise NotImplementedError

    def init_pop(self, pop_size):
        """Initialize the entire population

        Parameters:
        - pop_size: number of individuals

        Returns a list of individuals created by Toolkit.create"""
        return [self.create() for _ in range(pop_size)]

    def select(self, fit_and_pop):
        """Selects offspring from a population.

        Parameters:
        - fit_and_pop: list of (fitness, individual) tuples

        Returns a list of chosen (fitness, offspring_ind) tuples."""
        raise NotImplementedError

    def vary(self, pop):
        """Varies a population between the individuals in itself

        Parameters:
        - pop: population to be varied

        Returns a new, varied population."""
        raise NotImplementedError

    def fitness(self, ind):
        """Evaluated an individual's fitness.

        Parameters:
        - ind: individual to be evaluated.

        Returns a numerical value corresponding to its fitness."""
        raise NotImplementedError

    @staticmethod
    def map(*args, **kwargs):
        """Map used for evaluating fitnesses. May be replaced by a
        parallel map function, such as multiprocessing.Pool.map."""
        return map(*args, **kwargs)
