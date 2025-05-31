import argparse
from itertools import product

from numpy.linalg import eigvals
from lib.total_variation import TotalVariation
from models.lib.firstorderode.base import RungeKutta
from models.lib.firstorderode.lorenz import Lorenz63
from models.lib.firstorderode.roessler import Roessler76
from models.lib.firstorderode.sprott import SprottAttractors
from models.lib.toy_models import LinaJL
from reservoir_helper import get_lorenz63_data_and_reservoir

from lib.attractor_deviation import AttractorDeviation
from lib.attractor_inclusion import AttractorInclusion
from lib.correlation_integral import LongSpatial02CI, LongSpatialCI

parser = argparse.ArgumentParser()
parser.add_argument('--use_02_model', help="Choose between 0.02 and 0.1 model", action='store_true')
parser.add_argument('--use_uncoupled', help="Use uncoupled Model", action='store_true')
parser.add_argument("--n_nodes", type=int, help="Number of nodes for reservoir", default=20)
parser.add_argument("--n_samples", type=int, help="Number of different reservoir", default=100000)
parser.add_argument("--spectral_radius", type=float, default=0.4)
parser.add_argument("i_model", type=int, help="Choose which model to use")
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

import numpy as np
import matplotlib.pyplot as plt

from numpy import absolute

bounded = lambda pre: not (absolute(pre) > 2).any()
last = 5
osc = lambda pre, last=3: not (absolute(pre[-last:-1] - pre[-1]) < 5e-3).all() 

lcis = list(map(
    lambda x: CI(f"data/heikki_long{"_dt_02" if args.use_02_model else "" }/{x}_{CI.__name__}_ndata_300_nmul_100_seed_0.npz"), (
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

from multiprocessing import Pool
from numpy.random import default_rng
from tqdm import tqdm
from lib.rmse import rmse
from lib.valid_prediction_time import valid_prediction_time
from scipy.stats import chi2
from numpy.random import SeedSequence
collected = []

def worker(inputs):
    spectral_radius, i_model, seed = inputs
    measures = []
    names = []
    rng = default_rng(seed)

    if args.use_uncoupled:
        weights_internal = np.eye(args.n_nodes)
    else:
        weights_internal = rng.random((args.n_nodes, args.n_nodes))
        weights_internal /= (absolute(eigvals(weights_internal))).max()

    predictions_closed_loop, predictions_open_loop, data, normalizer, model, train, open_loop, closed_loop = get_lorenz63_data_and_reservoir(seed, args.n_nodes, models[i_model], spectral_radius=spectral_radius, regulizer=(1e-7 if args.use_02_model else 5e-6), weights_internal = weights_internal, g_in =(0.4 if args.use_02_model else 1) )
    
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
    measures += chiout[-1][0],
    names += "Heikki",

    measures += adevs[i_model](compare, normalizer=normalizer),
    measures += ddevs[i_model](compare, normalizer=normalizer),
    measures += aincs[i_model](compare, normalizer=normalizer),

    names += ("Adev", "Ddev", "AIncs")
            
    return measures, names

with Pool(48) as p:

    spectral_radius = args.spectral_radius

    i_model, Model = list(enumerate(models))[args.i_model]
    seed = 0
    n = args.n_samples
    ss = SeedSequence(seed)
    seeds = ss.spawn(n)

    entropies = np.array(list(map(lambda x: x.entropy, seeds)))
    

    collected = []

    inputs = list(zip([spectral_radius]*n, [i_model]*n, ss.spawn(n)))
    print(inputs[0])
    collected += list(zip(*tqdm(p.imap(worker, inputs), total=n, smoothing=0))),

    measures, names = list(zip(*collected))
    if args.use_uncoupled:
        name = "uncoupled"
    else:
        name = "random"

    from os import makedirs

    makedirs(f"results/spectral_radius_n_nodes_{args.n_nodes}", exist_ok = True) 
    np.savez(f"results/spectral_radius_n_nodes_{args.n_nodes}/{name}_single_{spectral_radius}{"_dt_02" if args.use_02_model else "" }_{Model(0,0).ode.__class__.__name__}.npz", measures = measures, names = names, entropies = entropies)
