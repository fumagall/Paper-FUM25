from typing import Any, Callable
from numpy import absolute, arctan, array, eye, mean, ndarray, save, savez, sign, std, zeros
from numpy.linalg import eigvals
from tqdm import tqdm
from lib.attractor_inclusion import AttractorInclusion
from lib.correlation_integral import SpatialCI
from lib.distribution_deviation import DistributionDeviation
from lib.local_step_ahead import LocalStepAheadPredictability
from lib.rmse import rmse
from lib.valid_prediction_time import valid_prediction_time
from models.lib.toy_models import LinaJL
from models.lib.train_test import UniformNormalizer
from reservoir.lib.reservoir import JAU24a as Reservoir
from numpy.random import default_rng

n_buffer = 10_000
n_train = 10_000
n_open_loop = 5_000 
n_closed_loop = 5_000 

def get_lorenz63_data_and_reservoir(seed, n_nodes, modelclass=LinaJL, weights_internal=None, spectral_radius = 0.44, regulizer = 1e-7, g_in = 1.0):
    model = modelclass(seed, n_buffer=n_buffer)

    rng = default_rng(seed=seed)

    unnormalized_data: ndarray = model.get_timeseries(3 * n_buffer + n_train + n_open_loop + n_closed_loop)[0]

    train = slice(n_buffer, n_buffer+n_train)
    open_loop = slice(n_buffer+train.stop, n_buffer+train.stop+n_open_loop)
    closed_loop = slice(n_buffer+open_loop.stop, n_buffer+open_loop.stop+n_closed_loop)
 
    normalizer = UniformNormalizer(unnormalized_data[train])
    data = normalizer(unnormalized_data)

    weights_input=rng.random((n_nodes, 3))
    if weights_internal is None:
        weights_internal = rng.random((n_nodes, n_nodes))
        weights_internal /= (absolute(eigvals(weights_internal))).max()
    reservoir = Reservoir(
        weights_internal=weights_internal, # type: ignore
        weights_input=weights_input, # type: ignore
        k = spectral_radius,
        g = g_in
    )

    responses = reservoir.driven_response(
        initial_reservoir_state=zeros(n_nodes),
        driving_inputs=data
    )

    weights_with_bias = reservoir.linear_regression_inv( 
        reservoir_responses=responses[train][:-1], # type: ignore
        target= data[train][1:],
        regulazation_lambda=regulizer
    )

    predictions_open_loop = reservoir.predict(
        reservoir_responses = responses[open_loop],
        weights_with_bias= weights_with_bias
    )

    predictions_closed_loop = reservoir.run_and_predict_autonomous_system(
        inital_reservoir_state = responses[closed_loop][0],
        weights_with_bias= weights_with_bias,
        n_timesteps=n_closed_loop
    )[1]

    return predictions_closed_loop, predictions_open_loop, data, normalizer, model, train, open_loop, closed_loop
