from multiprocessing.pool import Pool
from numpy.random import SeedSequence, default_rng
from tqdm import tqdm
from models.lib.firstorderode.base import RungeKutta
from models.lib.firstorderode.lorenz import Lorenz63
from models.lib.firstorderode.roessler import Roessler76
from models.lib.firstorderode.sprott import SprottAttractors
from models.lib.toy_models import LinaJL
import numpy as np
import matplotlib.pyplot as plt

Model = lambda a, b, c, seed: RungeKutta(Lorenz63(a, b, c, seed), 1e-3, 20, 10_000)

from lib.correlation_integral import LongSpatialCI

n_samples = 40_000
seed = 0
n_parallel = 24

ss = SeedSequence(seed)

def worker(seed):
    ci = LongSpatialCI("results/data/Lorenz63_LongSpatialCI_ndata_300_nmul_100_seed_0.npz")

    rng = default_rng(seed)

    theta = np.array((10, 28, 8/3))
    timeseries = Model(*theta, rng.integers(0, 2**31)).get_timeseries(5_000) # type: ignore
    p_y_log = -1/2*ci(timeseries[0], rng)[-1]

    mcmc_samples = []
    mcmc_acc = []
    mcmc_acc_sum = 0

    for _ in (range(n_samples)):
        proposal_theta = theta + rng.normal(0, 0.01, 3)
        proposal_theta_timeseries = Model(*proposal_theta, rng.integers(0, 2**31)).get_timeseries(5_000)[0]
        proposal_p_y_log = -1/2* ci(proposal_theta_timeseries, rng)[-1]
        alpha = np.exp(proposal_p_y_log - p_y_log)
        r = min(1 , alpha)
        mcmc_acc.append(r)
        if r > rng.random():
            theta = proposal_theta
            p_y_log = proposal_p_y_log
            mcmc_acc_sum += 1
        mcmc_samples.append(theta)

    mcmc_samples = np.array(mcmc_samples)
    return mcmc_samples, mcmc_acc_sum/n_samples

with Pool(48) as p:
    mcmc_samples, acceptance_rates = list(zip(*tqdm(p.imap(worker, ss.spawn(n_parallel)))))

print(np.array(mcmc_samples).shape, np.array(acceptance_rates).shape)

np.savez("results/mcmc_heikki_0.01.npz", samples = np.array(mcmc_samples), acceptance_rate=np.array(acceptance_rates))