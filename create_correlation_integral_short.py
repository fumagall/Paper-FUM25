import argparse
import sys
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, int64
import numba as nb
from numpy.random import Generator, default_rng, SeedSequence
from numpy import array, ndarray, sqrt
from multiprocessing import Pool
from models.lib.firstorderode.lorenz import Lorenz63
from models.lib.firstorderode.roessler import Roessler76
from models.lib.firstorderode.sprott import SprottAttractors
from numpy import fromiter
from tqdm import tqdm
from lib.correlation_integral import SpatialCI02, TransformedCorrelationIntegral, SpatialCI
from scipy.stats import chi2

from models.lib.firstorderode.base import RungeKutta
from models.lib.toy_models import JAU24a
from joblib import Parallel, delayed
from itertools import combinations
from multiprocessing import Pool
from multiprocessing import shared_memory

l = 5000 # how long the short timeseries is
n= 600 # how many timeseries to generate
mul = 1 # how much longer the long timeseries is compared to the short timeseries
seed = 0 

parser = argparse.ArgumentParser()
parser.add_argument('--use_02_model', help="Choose between 0.02 and 0.1 model", action='store_true')
parser.add_argument('i_model', type=int, help="Choose which model")
args = parser.parse_args()

if args.use_02_model:
    models = []
    for ODEClass in [* SprottAttractors[1:4], Roessler76]:
        models += lambda seed = None, n_buffer = 10000, ODEClass=ODEClass: RungeKutta(ODEClass(seed), 2e-2, 10, n_buffer),
    models += lambda seed = None, n_buffer = 10000: RungeKutta(Lorenz63(s=10, r=28, b=8 / 3, seed=seed,), odeint_dt=1e-3, step_size=20, n_buffer=n_buffer,),
    CI = SpatialCI02
else:
    models = []
    for ODEClass in [* SprottAttractors[1:4], Roessler76]:
        models += lambda seed = None, n_buffer = 10000, ODEClass=ODEClass: RungeKutta(ODEClass(seed), 2e-2, 50, n_buffer),
    models += lambda seed = None, n_buffer = 10000: RungeKutta(Lorenz63(s=10, r=28, b=8 / 3, seed=seed,), odeint_dt=1e-3, step_size=100, n_buffer=n_buffer,),
    CI = SpatialCI
Model = models[args.i_model]


fname = f"data/heikki_short{"_dt_02" if args.use_02_model else "" }/{Model().ode.__class__.__name__}_{CI.__name__}_ndata_{n}_nmul_{mul}_seed_{seed}" 
seed_sequence = SeedSequence(seed)

# create short timeseries, that get tested
def wrapper(seed):
    return CI.transform(Model(n_buffer=10000, seed=seed).get_timeseries(l, 1)[0])

with Pool(64) as p:
    o_timeseries = fromiter(tqdm(p.imap(wrapper, seed_sequence.spawn(n)), total=n), np.dtype((float, (l, 3))))

timeseries = o_timeseries

# create long timeseries as reference
def wrapper2(seed):
    return CI.transform(Model(n_buffer=10000, seed=seed).get_timeseries(mul*l, 1)[0])

with Pool(64) as p:
    o_long_timeseries = fromiter(tqdm(p.imap(wrapper2, seed_sequence.spawn(n)), total=n), np.dtype((float, (mul*l, 3))))
long_timeseries = o_long_timeseries

# Create shared memory to store timeseries
shm = shared_memory.SharedMemory(create=True, size=timeseries.nbytes)
shared_timeseries = np.ndarray(timeseries.shape, dtype=timeseries.dtype, buffer=shm.buf)
shared_timeseries[:] = timeseries[:]  # Copy data to shared memory

long_shm = shared_memory.SharedMemory(create=True, size=long_timeseries.nbytes)
shared_timeseries = np.ndarray(long_timeseries.shape, dtype=long_timeseries.dtype, buffer=long_shm.buf)
shared_timeseries[:] = long_timeseries[:]  # Copy data to shared memory

name3=long_shm.name
shape3=long_timeseries.shape
dtype3=long_timeseries.dtype

name=shm.name
shape=timeseries.shape
dtype=timeseries.dtype

# calculate min and maximum distance for each pair of test and reference timeseries
def correlation_sum_worker(arg):
    (seed, (i, j)) = arg
    rng = default_rng(seed)
    # Reconnect to shared memory
    existing_shm = shared_memory.SharedMemory(name=name)
    timeseries_shm = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    # return TransformedCorrelationIntegral.distance_minmax(CI.random_subset(rng, timeseries_shm[i]), CI.random_subset(rng, timeseries_shm[j]))
    existing_shm3 = shared_memory.SharedMemory(name=name3)
    long = np.ndarray(shape3, dtype=dtype3, buffer=existing_shm3.buf)

    return TransformedCorrelationIntegral.distance_minmax(CI.random_subset(rng, timeseries_shm[i]), CI.random_subset(rng, long[j]))

with Pool(64) as p:
    comb = combinations(range(n), 2)
    inputs = zip(seed_sequence.spawn(int(n*(n-1)/2)), comb)
    minmax = list(tqdm(p.imap(correlation_sum_worker, inputs, chunksize=1), total=int(n*(n-1)/2))) # type: ignore

# calculate bins for the radii
a = sqrt(max(array(minmax)[:,0]))
b = sqrt(min(array(minmax)[:,1]))
delta=(b-a)/100*2
a = a+delta
b = b-delta
N = 10+2
R0 = b
bb = (R0/a)**(1/(N-1))
bins=R0*bb**(-np.arange(0, N))[1:-1]

# calculate correlation sum for each pair of test and reference timeseries
def correlation_sum_worker2(arg):
    (seed, (i, j)) = arg
    rng = default_rng(seed)

    # Reconnect to shared memory
    existing_shm = shared_memory.SharedMemory(name=name)
    timeseries_shm = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    existing_shm3 = shared_memory.SharedMemory(name=name3)
    long = np.ndarray(shape3, dtype=dtype3, buffer=existing_shm3.buf)

    return TransformedCorrelationIntegral.correlation_sum_heikki(CI.random_subset(rng, timeseries_shm[i]), CI.random_subset(rng, long[j]), bins)

with Pool(64) as p:
    comb = combinations(range(n), 2)
    inputs = zip(seed_sequence.spawn(int(n*(n-1)/2)), comb)
    out = array(list(tqdm(p.imap(correlation_sum_worker2, inputs, chunksize=1), total=int(n*(n-1)/2)))) # type: ignore

# Clean up shared memory
shm.close()
shm.unlink()
long_shm.close()
long_shm.unlink()

# plot hist of edge radii
plt.figure()
plt.hist(out[:,0], bins=150)
plt.savefig(fname+"_0_histogram.pdf")
plt.figure()
plt.hist(out[:,-1], bins=150)
plt.savefig(fname+"_-1_histogram.pdf")

# save mean and cov
cov_subset, mean_subset = np.cov(out.T), np.mean(out.T, axis=1)
np.savez(fname, timeseries = long_timeseries[:], mean = mean_subset, cov = cov_subset, rs = bins) 

# plot chi2 test
ref_cov_inv = np.linalg.inv(cov_subset)
chi2l = list([
            (y - mean_subset) @ ref_cov_inv @ (y - mean_subset).T for y in out
        ])

plt.figure()
_, bins, _ = plt.hist(chi2l, bins=1450, density=True)
# Perform Chi-Square Test

x=np.linspace(min(bins), max(bins), 1000)
plt.xlim(0, 50)
plt.plot(x, chi2.pdf(x, df=len(mean_subset)))
plt.savefig(fname+"_chi2.pdf")   
