import numpy as np
from numpy import ndarray
from numpy.random import default_rng
from scipy.integrate import odeint

from .model import Model
from warnings import warn

class ODEIntLorenz63(Model):
    def __init__(self, s: float, r:  float, b: float, odeint_dt: float, step_size: int, n_buffer: int, seed: int | None):
        """Init Lorenz solved with odeint from scipy is used as integrator.

        dx/dt = s * (y - x),
        dy/dt = x * (r - z) - y
        dz/dt = x * y - b * z

        The effective timestep is calculated by odeint_dt * step_size

        :param float s: const for dx/dt.
        :param float r: const for dy/dt.
        :param float b: const for dz/dt.
        :param float odeint_dt: Lorenz steps in simulation.
        :param int step_size: Determines how many lorenz odeint_dt are between observations.
        :param int n_buffer: A buffer period that gets simulated at the beginning.
        :param int | None seed: The seed of the number genreator.
        """
        warn("This class was depricated and replaced by lib.odeint.lorenz.ODEIntLorenz63")
        self.s = s
        self.r = r
        self.b = b
        self.odeint_dt = odeint_dt
        self.step_size = step_size
        self.dt = self.odeint_dt * self.step_size
        self.rng = default_rng(seed)
        super().__init__(n_buffer)

    def sample_initial_states(self, size) -> ndarray:
        """
        A random prior distributed like Normal(( - 5.91652, - 5.52332, + 24.5723), 1)
        :param size: number of samplesriors (3 x size)
        :return: an array with p
        """
        return np.array((
            self.rng.normal(size=size) - 5.91652,
            self.rng.normal(size=size) - 5.52332,
            self.rng.normal(size=size) + 24.5723
        )).T

    @staticmethod
    def _dxdt(state, t, s, r, b):
        """
        Calculates the derivative of the Lorenz63 given the parameters.

        dx/dt = s * (y - x),
        dy/dt = x * (r - z) - y
        dz/dt = x * y - b * z

        :param state: A vector (x, y, z) with shape 3 x shape
        :param t: Time t (not used in Lorenz63 but important for odeint)
        :param s:
        :param r:
        :param b:
        :return: A vector with the derivative (x', y', z') of shape 3 x shape
        """
        x, y, z = state
        return (
            s * (y - x),
            x * (r - z) - y,
            x * y - b * z
        )

    def evolve_system(self, initial_state, timeseries_length):
        """
        Calculates the trajectory with odeint given the initial_state.
        It calculates the steps (0, num_of_dt_steps * dt, 2*num_of_dt_steps * dt, ..., num_of_dt_steps * (timeseries_length - 1) * dt).
        :param initial_state:
        :param timeseries_length:
        :return: Trajectory INCLUDING INITIAL STATE (x, y, z) of shape 3xnum_of_dt_steps
        """
        out = odeint(
            func=self._dxdt,
            y0=initial_state,
            t=self.odeint_dt * np.arange(0, int(self.step_size * timeseries_length)),
            args=(self.s, self.r, self.b)
        )
        return out[::self.step_size]
