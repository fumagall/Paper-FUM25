from abc import ABC, abstractmethod

from numpy import empty, ndarray
from numpy.random import Generator, default_rng


class Model(ABC):
    def __init__(self, n_buffer: int):
        self.n_buffer = n_buffer

    @abstractmethod
    def sample_initial_states(self, size: int = 1) -> ndarray:
        ...

    @abstractmethod
    def evolve_system(self, initial_states: ndarray, timeseries_length: int) -> ndarray:
        ...

    def get_timeseries(self, timeseries_length: int, size: int = 1) -> ndarray:
        """
        Sample size number of timeseries of length timeseries_length. With n_buffer steps, that don't get saved.

        :param timeseries_length:
        :param size:
        :param n_buffer:
        :param: sample_full:
        :return: The time series of size (size x timeseries_length x 3)
        """
        initial_state = self.sample_initial_states(size=size)

        states = empty((size, int(timeseries_length), *initial_state.shape[1:]))

        # run buffer
        for states_i in initial_state:
            states_i[:] = self.evolve_system(states_i, self.n_buffer + 1)[-1]

        # create output
        for result, initial in zip(states, initial_state):
            result[:] = self.evolve_system(initial, timeseries_length)

        return states

