import argparse
from lib.correlation_integral import LongSpatial02CI, LongSpatialCI
from lib.total_variation import TotalVariation
from models.lib.firstorderode.base import RungeKutta
from models.lib.firstorderode.lorenz import Lorenz63
from models.lib.firstorderode.roessler import Roessler76
from models.lib.firstorderode.sprott import SprottAttractors
from models.lib.toy_models import JAU24a, LinaJL

parser = argparse.ArgumentParser()
parser.add_argument('--use_02_model', help="Choose between 0.02 and 0.1 model", action='store_true')
parser.add_argument('--n_samples', help="Choose number of samples", type=int, default=50_000)
args = parser.parse_args()

if args.use_02_model:
    models = []
    for ODEClass in [* SprottAttractors[1:4], Roessler76]:
        models += lambda seed = None, n_buffer = 10000, ODEClass=ODEClass: RungeKutta(ODEClass(seed), 2e-2, 10, n_buffer),
    models += lambda seed = None, n_buffer = 10000: RungeKutta(Lorenz63(s=10, r=28, b=8 / 3, seed=seed,), odeint_dt=1e-3, step_size=20, n_buffer=n_buffer,),
    CI = LongSpatial02CI
else:
    models = []
    for ODEClass in [* SprottAttractors[1:4], Roessler76]:
        models += lambda seed = None, n_buffer = 10000, ODEClass=ODEClass: RungeKutta(ODEClass(seed), 2e-2, 50, n_buffer),
    models += lambda seed = None, n_buffer = 10000: RungeKutta(Lorenz63(s=10, r=28, b=8 / 3, seed=seed,), odeint_dt=1e-3, step_size=100, n_buffer=n_buffer,),
    CI = LongSpatialCI

from multiprocessing import Pool

from numpy.random import SeedSequence, default_rng
from tqdm import tqdm
from lib.attractor_deviation import AttractorDeviation
from lib.attractor_inclusion import AttractorInclusion

import numpy as np
import matplotlib.pyplot as plt

from lib.distribution_deviation import DistributionDeviation

seed = 0
n = args.n_samples

aincs = [ AttractorInclusion(f"data/hist/hist_{Model(0, 0).ode.__class__.__name__}.npz") for Model in models]
adevs = [ AttractorDeviation(f"data/hist/hist_{Model(0, 0).ode.__class__.__name__}.npz") for Model in models]
tvars = [ TotalVariation(f"data/hist/hist_{Model(0, 0).ode.__class__.__name__}.npz") for Model in models]
heikkis = [ CI(f"data/heikki_long{"_dt_02" if args.use_02_model else "" }/{Model(0, 0).ode.__class__.__name__}_{CI.__name__}_ndata_300_nmul_100_seed_0.npz") for Model in models]

def worker(seed):
    l = []
    rng = default_rng(seed)
    rng = rng.spawn(1)[0]
    for ainc, adev, tvar, heikki, Model in zip( aincs, adevs, tvars, heikkis, models ):
        timeseries = Model(seed).get_timeseries(5_000)
        l += np.array((
            ainc(timeseries[0]),
            adev(timeseries[0]),
            tvar(timeseries[0]),
            heikki(timeseries[0], rng=rng)[-1][0],
        )),
    return l

with Pool(64) as p:
    out = np.array(list(tqdm(p.imap(worker, SeedSequence(seed).spawn(n)), total=n, smoothing=0.01)))

np.save(f"results/reference_values_ranges{"_02" if args.use_02_model else "" }", out)

np.save(f"results/quantiles{"_02" if args.use_02_model else "" }", np.array((np.quantile(out, 0.0, axis=0), np.quantile(out, 0.95, axis=0))))