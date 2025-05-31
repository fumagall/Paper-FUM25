from numpy import array, ndarray
from .base import FirstOrderODE, FirstOrderSDE, ODEInt


class Lorenz63(FirstOrderODE):
    def __init__(self, s: float, r:  float, b: float, seed: int | None):
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
        self.s = s
        self.r = r
        self.b = b
        super().__init__(
            seed,
            s, 
            r,
            b
        )

    def sample_initial_states(self, size) -> ndarray:
        """
        A random prior distributed like Normal(( - 5.91652, - 5.52332, + 24.5723), 1)
        :param size: number of samplesriors (3 x size)
        :return: an array with p
        """
        return array((
            self.rng.normal(size=size) - 5.91652,
            self.rng.normal(size=size) - 5.52332,
            self.rng.normal(size=size) + 24.5723
        )).T

    @staticmethod
    def _dxdt(state, t, s, r, b) -> ndarray:
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
        return array((
            s * (y - x),
            x * (r - z) - y,
            x * y - b * z
        ))

class SDELorenz63(FirstOrderSDE):
    def __init__(self, sigma: float, s: float, r:  float, b: float, seed: int | None):
        """Init Lorenz solved with odeint from scipy is used as integrator.

        dx/dt = s * (y - x),
        dy/dt = x * (r - z) - y
        dz/dt = x * y - b * z

        The effective timestep is calculated by odeint_dt * step_size
        :param float sigma: const for noise.
        :param float s: const for dx/dt.
        :param float r: const for dy/dt.
        :param float b: const for dz/dt.
        :param float odeint_dt: Lorenz steps in simulation.
        :param int step_size: Determines how many lorenz odeint_dt are between observations.
        :param int n_buffer: A buffer period that gets simulated at the beginning.
        :param int | None seed: The seed of the number genreator.
        """
        self.sigma = sigma
        self.lorenz = Lorenz63(s, r, b, seed)
        super().__init__(
            seed,
            s, 
            r,
            b
        )

    def sample_initial_states(self, size) -> ndarray:
        return self.lorenz.sample_initial_states(size)

    def noise(self, state, t, *args) -> ndarray:
        return self.rng.normal(0, self.sigma, size=3)

    @staticmethod
    def _dxdt(state, t, *args) -> ndarray:
        return Lorenz63._dxdt(state, t, *args)

