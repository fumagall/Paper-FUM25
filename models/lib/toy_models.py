from numpy.random import default_rng

from .firstorderode.base import EulerMaruyama, RungeKutta

from .firstorderode.lorenz import Lorenz63, ODEInt, SDELorenz63


class JAU24a(RungeKutta):
    def __init__(self, seed: int | None = None, n_buffer=1000):
        """
        s=10,
        r=28,
        b=8 / 3
        effective_dt=0.1
        """
        super().__init__(
            Lorenz63(
                s=10,
                r=28,
                b=8 / 3,
                seed=seed,
            ),
            odeint_dt=1e-3,
            step_size=100,
            n_buffer=n_buffer,
        )


class LinaJL(ODEInt):
    def __init__(self, seed: int | None = None, n_buffer=1000):
        """
        Linas JupyterLab Model
        s=10,
        r=28,
        b=8 / 3
        effective_dt=0.02
        """
        super().__init__(
            Lorenz63(
                s=10,
                r=28,
                b=8 / 3,
                seed=seed,
            ),
            odeint_dt=1e-3,
            step_size=20,
            n_buffer=n_buffer,
        )

class SDELinaJL(EulerMaruyama):
    def __init__(self, sigma: float, seed: int | None = None, n_buffer=1000):
        sde = SDELorenz63(
                s=10,
                r=28,
                b=8 / 3,
                sigma=sigma,
                seed=seed,
            )
        super().__init__(
            sde,
            odeint_dt=1e-3,
            step_size=20,
            n_buffer=n_buffer,
        )


class LuciADevODEInt(ODEInt):
    def __init__(self, seed: int | None = None, n_buffer=1000):
        """
        Linas JupyterLab Model with step_size = 1
        s=10,
        r=28,
        b=8 / 3
        effective_dt=0.001
        """
        super().__init__(
            Lorenz63(
                s=10,
                r=28,
                b=8 / 3,
                seed=seed,
            ),
            odeint_dt=1e-3,
            step_size=1,
            n_buffer=n_buffer,
        )

class LuciADev(RungeKutta):
    def __init__(self, seed: int | None = None, n_buffer=1000):
        """
        Linas JupyterLab Model with step_size = 1
        s=10,
        r=28,
        b=8 / 3
        effective_dt=0.001
        """
        super().__init__(
            Lorenz63(
                s=10,
                r=28,
                b=8 / 3,
                seed=seed,
            ),
            odeint_dt=1e-3,
            step_size=1,
            n_buffer=n_buffer,
        )