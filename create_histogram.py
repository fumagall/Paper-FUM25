import argparse
from os import makedirs, mkdir
import sys
from numba import njit
from numpy import arange, array, histogramdd, load, max, min, save, savez, zeros
from numpy.random import SeedSequence
from lib.attractor_inclusion import AttractorInclusion
from models.lib.firstorderode.base import RungeKutta
from models.lib.firstorderode.lorenz import Lorenz63
from models.lib.firstorderode.roessler import Roessler76
from models.lib.firstorderode.sprott import SprottAttractors
from models.lib.model import Model
from models.lib.toy_models import LuciADev
from tqdm import tqdm

from multiprocessing import Pool
from multiprocessing import shared_memory

n_total = int(2e7)
n_minmax = n_total
n_jobs = 60
n_loops = 60
n_buf=16
n_bins = 256-2*n_buf
n_eff = 2*n_buf+n_bins

parser = argparse.ArgumentParser(description="Script to create the adev.")
        
# Add non-optional (positional) arguments
# Add optional arguments
parser.add_argument('--seed', default=0, type=int, help='If this value is set a simulation will executed with the given seed. Use collect to collect all seed into a single value (for cluster implementations)')
parser.add_argument('--num_cores', type=int, default=None, help='Specify number of Pool cores')

args = parser.parse_args()

models = []
for ODEClass in [* SprottAttractors[1:4], Roessler76]:
    models += lambda seed = None, n_buffer = 10000, ODEClass=ODEClass: RungeKutta(ODEClass(seed), 2e-2, 1, n_buffer),
models += lambda seed = None, n_buffer = 10000: RungeKutta(Lorenz63(s=10, r=28, b=8 / 3, seed=seed,), odeint_dt=1e-3, step_size=1, n_buffer=n_buffer,),

for ModelClass in models:
    ss = SeedSequence(args.seed)

    # create common minmax
    def minmax_wrapper(seed):
        timeseries = ModelClass(seed).get_timeseries(n_minmax)[0]
        return min(timeseries, axis=0), max(timeseries, axis=0)
    
    with Pool(args.num_cores) as p:
        minmaxs = array(list(p.map(minmax_wrapper, ss.spawn(n_jobs))))

    mi, ma = (
            min(minmaxs[:, 0], axis=0),
            max(minmaxs[:, 1], axis=0)
        )
    
    print(mi, ma)

    n_eff = n_buf*2+n_bins

    def adev_wrapper(seed):
        adev = zeros((n_eff, n_eff, n_eff), dtype=int)
        model = ModelClass(seed, n_buffer=int(1e7))

        for _ in range(n_loops):
            timeseries = model.get_timeseries(n_total)[0]
            AttractorInclusion.add_timeseries_to_histogram(adev, timeseries, mi, ma, n_buf, n_bins)
        
        return adev
    
    adev = zeros((n_eff, n_eff, n_eff), dtype=int)
    with Pool(args.num_cores) as p:
        for adev_i in p.imap(adev_wrapper, ss.spawn(n_jobs)):
            adev += adev_i

    savez(f"data/hist/hist_{ModelClass().ode.__class__.__name__}.npz", 
          hist=adev, 
          minimum=mi, 
          maximum=ma, 
          n_minmax = n_minmax,
          seed = args.seed,
          n_jobs = n_jobs,
          n_loops = n_loops,
          n_buf=n_buf,
          n_bins = n_bins,
          n_eff = n_eff,
        )