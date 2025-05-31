import unittest

from numpy import all
from numpy.random import default_rng

from ..lib.train_test import split_and_normalize_uniformly

from ..lib.toy_models import JAU24a
from ..lib.firstorderode.lorenz import ODEInt, Lorenz63
NewLorenz63 = lambda s, r, b, odeint_dt, step_size, n_buffer, seed: ODEInt(Lorenz63(s, r, b, seed), odeint_dt, step_size, n_buffer)
from ..lib.odeint_lorenz63 import ODEIntLorenz63 as OldLorenz63


class MyTestCase(unittest.TestCase):
    def test_JAU24a(self):
        l = JAU24a()
        t = l.get_timeseries(7000, size=3)
        train, test, normalizer = split_and_normalize_uniformly(t, 1000)
        assert(all(test - normalizer(test)))
        assert(all(train - normalizer(train)))
        assert(test.shape[1] == 1000)
        assert(train.shape[1] == 7000-1000)
        print(t.shape)
        l = JAU24a(0)
        t = l.get_timeseries(7000).squeeze()
        assert(t.shape[0] == 7000)


    def test_old_vs_new(self):
        rng = default_rng()
        seed = rng.integers(0, 2**32)
        odeint_dt = rng.random()*1e-2
        step_size = rng.integers(0, 30)
        n_buffer = rng.integers(0, 1000)
        size = rng.integers(1, 5)
        tlen = rng.integers(100, 10000)

        l = JAU24a(seed=seed)
        nl = NewLorenz63(10,
            28,
            8 / 3,odeint_dt,
            step_size,
            n_buffer,
            seed )
        ol = OldLorenz63(10,
            28,
            8 / 3,odeint_dt,
            step_size,
            n_buffer,
            seed )
        assert(all(ol.get_timeseries(tlen, size) == nl.get_timeseries(tlen, size)))

if __name__ == '__main__':
    unittest.main()
