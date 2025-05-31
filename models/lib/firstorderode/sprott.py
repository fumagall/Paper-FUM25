from warnings import warn
from numpy import array, ndarray
from .base import FirstOrderODE

from numpy import array, ndarray
from .base import FirstOrderODE

class FirstOrderODESprottA(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        warn("Be carful when randomely choosing the initial conditions for the Sprott A System. The system can be an volume conserving system depending on the initial conditions DOI: 10.1140/epjst/e2015-02472-1")
        std = 0.01
        mean = array((
            0,
            5,
            0
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            y,
            -x + y*z,
            1-y**2
        ))

class FirstOrderODESprottB(FirstOrderODE):
    def __init__(self, seed: int | None = None, a=1.0, b=1.0, c=1.0):
        """Sprott B System

        from 

        :param int | None seed: for initial condition, defaults to None
        :param float a: , defaults to 1.0
        :param float b: , defaults to 1.0
        :param float c: , defaults to 1.0
        """
        super().__init__(seed, a, b, c)

    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            0.05,
            0.05,
            0.05
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t, a, b, c):
        x, y, z = state
        return array((
            a*y*z,
            x-b*y,
            c-x*y
        ))
    
class FirstOrderODESprottC(FirstOrderODE):
    def __init__(self, seed: int | None = None, a=1.0, b=1.0, c=1.0):
        """Sprott C System

        from https://link.springer.com/article/10.1007/s11071-011-0235-8

        :param int | None seed: for initial condition, defaults to None
        :param float a: , defaults to 1.0
        :param float b: , defaults to 1.0
        :param float c: , defaults to 1.0
        """
        super().__init__(seed, a, b, c)

    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            0.05,
            0.05,
            0.05
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t, a, b, c):
        x, y, z = state
        return array((
            y*z,
            x - y,
            1-x**2
        ))
    
class FirstOrderODESprottD(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            -2,
            -1,
            1
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            -y,
            x+z,
            x*z + 3 * y**2
        ))
    
class FirstOrderODESprottE(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            1.5,
            1,
            0
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            y*z,
            x**2 - y,
            1 - 4*x
        ))
    
class FirstOrderODESprottF(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            0,
            0,
            0.5
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            y + z,
            -x + 0.5 * y,
            x**2 - z
        ))
    
class FirstOrderODESprottG(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            -0.5,
            -0.5,
            0
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            0.4 * x + z,
            x*z - y,
            -x + y
        ))
    
class FirstOrderODESprottH(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            0.05,
            0.05,
            0.05
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            -y + z**2,
            x+ 0.5*y,
            x-z
        ))
    
class FirstOrderODESprottI(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.001
        mean = array((
            -0.2,
            -0.5,
            0
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            -0.2*y,
            x+z,
            x+y**2-z
        ))
    
class FirstOrderODESprottJ(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.05
        mean = array((
            7.5,
            -3,
            -10
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            2*z,
            -2*y + z,
            -x+y+y**2
        ))

class FirstOrderODESprottK(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            -1.5,
            -0.5,
            2
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            x*y-z,
            x-y,
            x+0.3*z
        ))
    
class FirstOrderODESprottL(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            5,
            25,
            -9
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            y+3.9*z,
            0.9*x**2-y,
            1-x
        ))
    
class FirstOrderODESprottM(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            2.5,
            -4.5,
            0
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            -z,
            -x**2-y,
            1.7+1.7*x+y
        ))
    
class FirstOrderODESprottN(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            -15,
            -5,
            0
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            -2*y,
            x+z**2,
            1 + y -2*z
        ))
    
class FirstOrderODESprottO(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.001
        mean = array((
            -0.2,
            -0.4,
            0
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            y,
            x-z,
            x+x*z + 2.7*y
        ))
    
class FirstOrderODESprottP(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            0.05,
            0.05,
            0.05
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            2.7*y + z,
            -x + y**2,
            x + y
        ))
    
class FirstOrderODESprottQ(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            -2,
            -2,
            -2
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            -z,
            x-y,
            3.1*x+y**2+0.5*z
        ))
    
class FirstOrderODESprottR(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            -3,
            -2,
            -1
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            0.9 - y,
            0.4 + z,
            x*y - z
        ))
    
class FirstOrderODESprottS(FirstOrderODE):
    def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            0.05,
            0.05,
            0.05
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]

    @staticmethod
    def _dxdt(state, t):
        x, y, z = state
        return array((
            -x - 4*y,
            x + z**2,
            1+x
        ))
    
SprottAttractors: list[type[FirstOrderODE]] = [
    FirstOrderODESprottA,
    FirstOrderODESprottB,
    FirstOrderODESprottC,
    FirstOrderODESprottD,
    FirstOrderODESprottE,
    FirstOrderODESprottF,
    FirstOrderODESprottG,
    FirstOrderODESprottH,
    FirstOrderODESprottI,
    FirstOrderODESprottJ,
    FirstOrderODESprottK,
    FirstOrderODESprottL,
    FirstOrderODESprottM,
    FirstOrderODESprottN,
    FirstOrderODESprottO,
    FirstOrderODESprottP,
    FirstOrderODESprottQ,
    FirstOrderODESprottR,
    FirstOrderODESprottS
]