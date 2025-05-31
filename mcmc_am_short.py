import argparse
from multiprocessing.pool import Pool
import sys
from typing import Callable
from numpy.random import SeedSequence, default_rng
from tqdm import tqdm
from models.lib.firstorderode.base import RungeKutta
from models.lib.firstorderode.lorenz import Lorenz63
from models.lib.firstorderode.roessler import Roessler76
from models.lib.firstorderode.sprott import SprottAttractors
from models.lib.toy_models import JAU24a, LinaJL
import numpy as np
import matplotlib.pyplot as plt

from lib.correlation_integral import LongSpatialCI, SpatialCI02 as SCI02, SpatialCI as SCI, TransformedCorrelationIntegral # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument('--use_02_model', help="Choose between 0.02 and 0.1 model", action='store_true')
parser.add_argument('--n_samples', help="Number of samples that should be created", type=int, default=30_000)
parser.add_argument('sigma', help="Varianze of the proposal kernel", type=float)
args = parser.parse_args()

if args.use_02_model:
    def Model(a, b, c, seed, n_buffer = 10000):
        return RungeKutta(Lorenz63(a, b, c, seed), odeint_dt=1e-3, step_size=20, n_buffer=n_buffer)
    
    class SpatialCI02(SCI02):
        def __call__(self, timeseries, rng: np.random.Generator, unnormalizer: Callable[[np.ndarray], np.ndarray] | None = None):

            def worker(timeseries, r_timeseries):
                timeseries = self.random_subset(rng, timeseries)
                r_timeseries = self.random_subset(rng, r_timeseries)
                y =  self.correlation_sum_heikki(timeseries, r_timeseries, self.rs)
                return (y - self.ref_mean) @ self.ref_cov_inv @ (y - self.ref_mean).T       
            
            ref_timeseries = self.ref_timeseries[rng.integers(0, len(self.ref_timeseries))] # type: ignore

            if unnormalizer is not None:
                timeseries = unnormalizer(timeseries)

            timeseries = self.transform(timeseries)

            chi2_sum = worker(timeseries, ref_timeseries)

            return chi2_sum,
    CI = SpatialCI02
else:
    def Model(a, b, c, seed, n_buffer = 10000):
        return RungeKutta(Lorenz63(a, b, c, seed), odeint_dt=1e-3, step_size=100, n_buffer=n_buffer)
    
    class SpatialCI(SCI):
        def __call__(self, timeseries, rng: np.random.Generator, unnormalizer: Callable[[np.ndarray], np.ndarray] | None = None):

            def worker(timeseries, r_timeseries):
                timeseries = self.random_subset(rng, timeseries)
                r_timeseries = self.random_subset(rng, r_timeseries)
                y =  self.correlation_sum_heikki(timeseries, r_timeseries, self.rs)
                return (y - self.ref_mean) @ self.ref_cov_inv @ (y - self.ref_mean).T       
            
            ref_timeseries = self.ref_timeseries[rng.integers(0, len(self.ref_timeseries))] # type: ignore

            if unnormalizer is not None:
                timeseries = unnormalizer(timeseries)

            timeseries = self.transform(timeseries)

            chi2_sum = worker(timeseries, ref_timeseries)

            return chi2_sum,
    CI = SpatialCI

n_samples = 30_000
seed = 0
t_0 = 50
n_parallel = 24
sig = args.sigma

ss = SeedSequence(seed)


def worker(inputs):
    seed, id = inputs
    samples = tqdm(range(n_samples)) if id == 0 else range(n_samples)
    #https://projecteuclid.org/journals/bernoulli/volume-7/issue-2/An-adaptive-Metropolis-algorithm/bj/1080222083.full
    #https://www2.stat.duke.edu/courses/Fall21/sta601.001/slides/09-adaptive-metropolis.html#48
    
    ci = CI(f"data/heikki_short{"_dt_02" if args.use_02_model else "" }/Lorenz63_{CI.__name__}_ndata_600_nmul_1_seed_0.npz")
    rng = default_rng(seed)

    theta = np.array((10, 28, 8/3))
    timeseries = Model(*theta, rng.integers(0, 2**31)).get_timeseries(5_000) # type: ignore
    p_y_log = -1/2*ci(timeseries[0], rng)[-1]

    mcmc_samples = []
    mcmc_acc = []
    mcmc_acc_sum = 0

    sigma = sig*np.eye(3)
    eps = 0
    s_d = (2.4)**2 /3

    for t in samples:
        if t == t_0:
            mean_old = np.mean(np.array(mcmc_samples), axis=0)
        if t > t_0:
            mean = np.mean(np.array(mcmc_samples), axis=0)
            sigma = (t-1)/t*sigma + s_d / t * (t * np.outer(mean_old, mean_old) - (t+1) * np.outer(mean, mean) + np.outer(mcmc_samples[-1], mcmc_samples[-1]) + eps * np.eye(3))
            mean_old = mean
        proposal_theta = theta + rng.multivariate_normal(np.zeros(3), sigma)
        proposal_theta_timeseries = Model(*proposal_theta, rng.integers(0, 2**31)).get_timeseries(5_000)[0] # type: ignore
        proposal_y = -1/2* ci(proposal_theta_timeseries, rng)[-1]
        alpha = np.exp(proposal_y - p_y_log)
        r = min(1 , alpha)
        mcmc_acc.append(r)
        if r > rng.random():
            theta = proposal_theta
            p_y_log = proposal_y
            mcmc_acc_sum += 1
        mcmc_samples.append(theta)

    mcmc_samples = np.array(mcmc_samples)
    return mcmc_samples, mcmc_acc_sum/n_samples

with Pool(n_parallel) as p:
    mcmc_samples, acceptance_rates = list(zip(*tqdm(p.imap(worker, zip(ss.spawn(n_parallel), range(n_parallel))))))

print(np.array(mcmc_samples).shape, np.array(acceptance_rates).shape)

np.savez(f"results/mcmc/heikki_short{"_dt_02" if args.use_02_model else "" }_{sig:.4f}.npz", samples = np.array(mcmc_samples), acceptance_rate=np.array(acceptance_rates))