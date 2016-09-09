class Toolkit:
    def create(self):
        raise NotImplementedError

    def init_pop(self, pop_size):
        return [self.create() for _ in range(pop_size)]

    def select(self, fit_and_pop):
        raise NotImplementedError

    def vary(self, pop):
        raise NotImplementedError

    def fitness(self, ind):
        raise NotImplementedError

    def map(*args, **kwargs):
        return map(*args, **kwargs)
