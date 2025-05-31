from numpy import array, ndarray
from ..firstorderode.base import FirstOrderODE


class Roessler76(FirstOrderODE):
   def __init__(self, seed: int | None = None, a=0.2, b=0.2, c=5.7):
      """The Roessler system:

         dx/dt = -(y + z),
         dy/dt = x + a * y,
         dz/dt =b + z*(x-c)

      from https://link.springer.com/book/10.1007/b97624 chapter 12.3

      :param int | None seed: for initial conditions, defaults to None
      :param float a: , defaults to 0.2
      :param float b: , defaults to 0.2
      :param float c: , defaults to 5.7
      """
      super().__init__(seed, a, b, c)

   def sample_initial_states(self, size) -> ndarray:
        std = 0.01
        mean = array((
            -7.8366,
            -4.1703,
            0.014385
        ))
        return self.rng.normal(size=(size, 3)) * std + mean[None, :]


   @staticmethod
   def _dxdt(state, t, a, b, c):
      x, y, z = state
      return (
         -(y + z),
         x + a * y,
         b + z*(x-c)
      )
 