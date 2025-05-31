import os
import sys
from lib.correlation_integral import LongSpatial02CI
from models.lib.firstorderode.base import RungeKutta
from models.lib.firstorderode.lorenz import Lorenz63
from models.lib.firstorderode.roessler import Roessler76
from models.lib.firstorderode.sprott import SprottAttractors
from models.lib.toy_models import LinaJL
from reservoir_helper import get_lorenz63_data_and_reservoir

dt_02 = True
n_nodes = 10
i_model = 4

#get_lorenz63_data_and_reservoir = lambda seed, n_nodes: get_lorenz63_data_and_reservoir(seed, n_nodes, Model)
import numpy as np
import matplotlib.pyplot as plt

from lib.attractor_deviation import AttractorDeviation
from lib.attractor_inclusion import AttractorInclusion
from lib.correlation_integral import LongSpatialCI, SpatialCI
from lib.distribution_deviation import DistributionDeviation
from lib.local_step_ahead import LocalStepAheadPredictability
from lib.total_variation import TotalVariation

if dt_02:
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


from numpy import absolute

bounded = lambda pre: not (absolute(pre) > 2).any()
last = 5
osc = lambda pre, last=3: not (absolute(pre[-last:-1] - pre[-1]) < 5e-3).all() 

from lib.attractor_deviation import AttractorDeviation
from lib.attractor_inclusion import AttractorInclusion
from lib.correlation_integral import LongSpatialCI, SpatialCI
from lib.distribution_deviation import DistributionDeviation
from lib.local_step_ahead import LocalStepAheadPredictability

lcis = list(map(
    lambda x: CI(f"data/heikki_long{"_dt_02" if dt_02 else ""}/" + x + f"_{CI.__name__}_ndata_300_nmul_100_seed_0.npz"), (
        map(lambda x: x().ode.__class__.__name__, models)
    ) 
))

adevs = list(map(
    lambda x: AttractorDeviation("data/hist/hist_" + x + ".npz"), (
        map(lambda x: x().ode.__class__.__name__, models)
    ) 
))

ddevs = list(map(
    lambda x: TotalVariation("data/hist/hist_" + x + ".npz"), (
        map(lambda x: x().ode.__class__.__name__, models)
    ) 
))

aincs = list(map(
    lambda x: AttractorInclusion("data/hist/hist_" + x + ".npz"), (
        map(lambda x: x().ode.__class__.__name__, models)
    ) 
))

plot_hist_measure = list(map(
    lambda x: TotalVariation("data/hist/hist_" + x + ".npz", resolution=64), (
        map(lambda x: x().ode.__class__.__name__, models)
    ) 
))

from multiprocessing import Pool
from numpy.random import default_rng
from tqdm import tqdm
from lib.rmse import rmse
from lib.valid_prediction_time import valid_prediction_time
from scipy.stats import chi2
from numpy.random import SeedSequence

if dt_02:
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

def get_trajectory(inputs):
    spectral_radius, i_model, seed = inputs
    measures = []
    measures_ref = []
    names = []
    rng = default_rng(seed)

    weights_internal = np.eye(n_nodes)


    predictions_closed_loop, predictions_open_loop, data, normalizer, model, train, open_loop, closed_loop = get_lorenz63_data_and_reservoir(seed, n_nodes, models[i_model], spectral_radius=spectral_radius, regulizer=0*(1e-7 if dt_02 else 5e-6), weights_internal = weights_internal, g_in =(0.4 if dt_02 else 1) )
    
    compare = predictions_closed_loop[:-1]
        
    measures += [
    bounded(predictions_closed_loop), # type: ignore
    osc(predictions_closed_loop),
    (valid_prediction_time(data[closed_loop][1:], predictions_closed_loop[:-1])),
    rmse(data[closed_loop][1:], predictions_closed_loop[:-1])**2
    ]

    names += [
    ("bounded"), # type: ignore
    ("ozillating"),
    ("valid_prediciton_time"),
    ("mse")
    ]

    chiout = lcis[i_model](compare, rng, unnormalizer=normalizer.unnormalize)

    measures += 1-aincs[i_model](compare, normalizer=normalizer),
    measures += adevs[i_model](compare, normalizer=normalizer),
    measures += ddevs[i_model](compare, normalizer=normalizer),

    measures += chiout[-1][0],
    names += "Heikki",

    measures_ref += 1-aincs[i_model](data[closed_loop], normalizer=normalizer),
    measures_ref += adevs[i_model](data[closed_loop], normalizer=normalizer),
    measures_ref += ddevs[i_model](data[closed_loop], normalizer=normalizer),
    chiout = lcis[i_model](data[closed_loop], rng, unnormalizer=normalizer.unnormalize)
    measures_ref += chiout[-1][0],
    

    names += ("AIncs", "Adev", "Ddev")
            
    return predictions_closed_loop, data[closed_loop], measures, measures_ref

models2 = [*SprottAttractors[1:4], Roessler76, Lorenz63]

data20 = np.load(f"results/spectral_radius_n_nodes_{20}/uncoupled_single_0.4{"" if not dt_02 else "_dt_02"}_{models2[i_model].__name__}.npz")
data20["measures"].shape

ss = SeedSequence(0)
ss.spawn(data20["measures"].shape[1])
ss.
seeds = np.array(ss.spawn(data20["measures"].shape[1]))

data = data20["measures"].squeeze()#
print(data.shape)
idx = np.all(data[:, :2] == 1, axis=-1)
data = data[idx, 4:]
data = data[:, [3, 1, 2, 0]]
data[:, 0] = 1- data[:, 0]
seeds = seeds[idx]
print(data.shape)

n = int(sys.argv[1])
def worker(seed):
    return get_trajectory((0.4, 4, seed))

index = 0
succesful_tries = []
pbar = tqdm(total=n, smoothing=0)


with Pool(48) as p:
    while len(succesful_tries) < n:
        for data in p.map(worker, ss.spawn(48)):
            if np.array(data[2][:2]).all() and len(succesful_tries) < n:
                succesful_tries.append(data)
                pbar.update(1)
        index += 48

all_rejected_measures = list(map(np.array, zip(*succesful_tries)))
pbar.close()

np.savez("results/create_figure_y", predictions_closed_loop = all_rejected_measures[0], data_closed_loop=all_rejected_measures[1], measures = all_rejected_measures[2], measures_ref = all_rejected_measures[3])