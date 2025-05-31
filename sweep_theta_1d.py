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
parser.add_argument('--n_threads', help="Choose number of threads", type=int, default=64)
args = parser.parse_args()

if args.use_02_model:
    def Model(a, b, c, seed, n_buffer = 10000):
        return RungeKutta(Lorenz63(a, b, c, seed), odeint_dt=1e-3, step_size=20, n_buffer=n_buffer)
    
    CI = LongSpatial02CI
else:
    def Model(a, b, c, seed, n_buffer = 10000):
        return RungeKutta(Lorenz63(a, b, c, seed), odeint_dt=1e-3, step_size=100, n_buffer=n_buffer)
    
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

ainc = AttractorInclusion(f"data/hist/hist_{Model(0,0,0,0, 0).ode.__class__.__name__}.npz") 
adev = AttractorDeviation(f"data/hist/hist_{Model(0,0,0,0, 0).ode.__class__.__name__}.npz")
tvar = TotalVariation(f"data/hist/hist_{Model(0,0,0,0, 0).ode.__class__.__name__}.npz")
heikki = CI(f"data/heikki_long{"_dt_02" if args.use_02_model else "" }/{Model(0,0,0,0, 0).ode.__class__.__name__}_{CI.__name__}_ndata_300_nmul_100_seed_0.npz")

xlims = np.array([[6.58, 14.9], [25.33, 30.5], [2.24, 3.18]])
n_linspace = 1000
thetas = np.array([np.linspace(*lim, n_linspace) for lim in xlims])

def worker(input):
    dim, seed = input
    l = []
    theta_ref = np.array((10, 28, 8/3))

    rng = default_rng(seed)
    rng = rng.spawn(1)[0]
    
    for theta_dim in thetas[dim]:
        theta = theta_ref.copy()
        theta[dim] = theta_dim
        timeseries = Model(*theta, seed).get_timeseries(5_000)[0]

        l += np.array((
            ainc(timeseries),
            adev(timeseries),
            tvar(timeseries),
            heikki(timeseries, rng=rng)[-1][0],
        )),
    return l

with Pool(args.n_threads) as p:
    out = np.array(list(tqdm(p.imap(worker, zip(np.repeat(range(3), n), SeedSequence(seed).spawn(n*3))), total=n*3))).reshape(3, n, n_linspace, 4)

xlims = [[6.58, 14.9], [25.33, 30.5], [2.24, 3.18]]
np.savez(f"results/sweeped_thetas{"_02" if args.use_02_model else "" }", data=out, thetas=thetas)