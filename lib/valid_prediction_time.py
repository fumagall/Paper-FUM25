from numba import jit, njit
from numpy import array, empty, empty_like, ndarray
from numpy.linalg import norm

@njit
def valid_prediction_time(reference: ndarray, prediction: ndarray, threshold: float = 0.4) -> int | None:
    """Calculates the valid prediction time: The first time where ||reference - prediction||^2 / ( <||reference - <reference>||^2>_time ) is larger then the threshold, where <.> is the average over time and ||.|| the norm over space.

    :param ndarray reference: NSamples x NDim. The reference timeseries
    :param ndarray prediction: NSamples x NDim. The prediction timeseries
    :param float threshold: _description_, defaults to 0.4
    :return int | None: Index where the condition first is met, None if there is no valid 
    """
    difference = empty(reference.shape[0])
    means = empty(reference.shape[1])
    if not array(reference.shape == prediction.shape).all():
        raise Exception("Reference and prediction do not have the same shape")

    for i in range(len(means)):
        means[i] = reference[:, i].mean()

    for i in range(len(reference)):
        difference[i] = norm(reference[i] - means)**2

    denominator = difference.mean()

    for i, (ref, pre) in enumerate(zip(reference[1:], prediction[1:])):
        if norm(ref - pre)**2 / denominator > threshold:
            return i+1
    
    return None
