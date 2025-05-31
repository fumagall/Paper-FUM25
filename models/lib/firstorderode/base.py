from abc import ABC, abstractmethod

from matplotlib.pylab import default_rng
from numba import njit
from numpy import arange, ndarray, zeros
from scipy.integrate import odeint, solve_ivp
from ..model import Model

class FirstOrderODE(ABC):
    def __init__(self, seed: int | None = None, *args):
        self.rng = default_rng(seed)
        self.args = args

    @abstractmethod
    def sample_initial_states(self, size) -> ndarray:
        pass

    @staticmethod
    @abstractmethod
    def _dxdt(state, t, *args) -> ndarray:
        pass

class FirstOrderSDE(ABC):
    def __init__(self, seed: int | None = None, *args):
        self.rng = default_rng(seed)
        self.args = args

    @abstractmethod
    def sample_initial_states(self, size) -> ndarray:
        pass

    @abstractmethod
    def noise(self, state, t, *args) -> ndarray:
        pass

    @staticmethod
    @abstractmethod
    def _dxdt(state, t, *args) -> ndarray:
        pass

class RungeKutta(Model):
    def __init__(self, ode: FirstOrderODE, odeint_dt: float, step_size: int, n_buffer: int):
        """Init model solved with odeint from scipy is used as integrator.
        The effective timestep is calculated by odeint_dt * step_size.
        
        :param float odeint_dt: Lorenz steps in simulation.
        :param int step_size: Determines how many lorenz odeint_dt are between observations.
        :param int n_buffer: A buffer period that gets simulated at the beginning.
        :param int | None seed: The seed of the number genreator.
        """
        self.odeint_dt = odeint_dt
        self.ode = ode
        self.step_size = step_size
        self.dt = self.odeint_dt * self.step_size
        super().__init__(n_buffer)

    def evolve_system(self, initial_state, timeseries_length):
        """
        Calculates the trajectory with odeint given the initial_state.
        It calculates the steps (0, num_of_dt_steps * dt, 2*num_of_dt_steps * dt, ..., num_of_dt_steps * (timeseries_length - 1) * dt).
        :param initial_state:
        :param timeseries_length:
        :return: Trajectory INCLUDING INITIAL STATE (x, y, z) of shape 3xnum_of_dt_steps
        """
        def dxdt(t, y, *args):
            return self.ode._dxdt(y, t, *args)
                                  
        out = solve_ivp(
            fun=dxdt,
            y0=initial_state,
            t_span = (0, self.odeint_dt * (self.step_size * (timeseries_length) + 1)),
            t_eval=self.odeint_dt * arange(0, int(self.step_size * timeseries_length)),
            args=self.ode.args,
            method="RK45"
        ).y.T
        return out[::self.step_size]

    def sample_initial_states(self, size: int = 1) -> ndarray:
        return self.ode.sample_initial_states(size)
    

class ODEInt(Model):
    def __init__(self, ode: FirstOrderODE, odeint_dt: float, step_size: int, n_buffer: int):
        """Init model solved with odeint from scipy is used as integrator.
        The effective timestep is calculated by odeint_dt * step_size.
        
        :param float odeint_dt: Lorenz steps in simulation.
        :param int step_size: Determines how many lorenz odeint_dt are between observations.
        :param int n_buffer: A buffer period that gets simulated at the beginning.
        :param int | None seed: The seed of the number genreator.
        """
        self.odeint_dt = odeint_dt
        self.ode = ode
        self.step_size = step_size
        self.dt = self.odeint_dt * self.step_size
        super().__init__(n_buffer)

    def evolve_system(self, initial_state, timeseries_length):
        """
        Calculates the trajectory with odeint given the initial_state.
        It calculates the steps (0, num_of_dt_steps * dt, 2*num_of_dt_steps * dt, ..., num_of_dt_steps * (timeseries_length - 1) * dt).
        :param initial_state:
        :param timeseries_length:
        :return: Trajectory INCLUDING INITIAL STATE (x, y, z) of shape 3xnum_of_dt_steps
        """
        out = odeint(
            func=self.ode._dxdt,
            y0=initial_state,
            t=self.odeint_dt * arange(0, int(self.step_size * timeseries_length)),
            args=self.ode.args
        )
        return out[::self.step_size]

    def sample_initial_states(self, size: int = 1) -> ndarray:
        return self.ode.sample_initial_states(size)
    

class EulerMaruyama(Model):
    def __init__(self, sde: FirstOrderSDE, odeint_dt: float, step_size: int, n_buffer: int):
        """Init model solved with euler maruyama as integrator.
        The effective timestep is calculated by odeint_dt * step_size.
        
        :param FirstOrderSDE sde: The SDE model.
        :param float odeint_dt: Lorenz steps in simulation.
        :param int step_size: Determines how many lorenz odeint_dt are between observations.
        :param int n_buffer: A buffer period that gets simulated at the beginning.
        """

        self.model = sde
        self.odeint_dt = odeint_dt
        self.step_size = step_size
        self.dt = self.odeint_dt * self.step_size
        super().__init__(n_buffer)

    def sample_initial_states(self, size: int = 1) -> ndarray:
        return self.model.sample_initial_states(size)

    def evolve_system(self, initial_state, timeseries_length):
        """
        Calculates the trajectory with odeint given the initial_state.
        It calculates the steps (0, num_of_dt_steps * dt, 2*num_of_dt_steps * dt, ..., num_of_dt_steps * (timeseries_length - 1) * dt).
        :param initial_state:
        :param timeseries_length:
        :return: Trajectory INCLUDING INITIAL STATE (x, y, z) of shape 3xnum_of_dt_steps
        """
        out = zeros((timeseries_length, 3))
        out[0] = initial_state
        for i in range(timeseries_length-1):
            tmp = out[i]
            for j in range(self.step_size):
                time = self.odeint_dt * (i * self.step_size + j)
                tmp = self.model._dxdt(tmp, time, *self.model.args) * self.odeint_dt + self.model.noise(tmp, time, *self.model.args)
            out[i+1] = tmp
        return out




