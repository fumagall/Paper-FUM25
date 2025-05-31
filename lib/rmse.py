from numpy import mean, ndarray, sqrt


def rmse(reference: ndarray, prediction: ndarray) -> float:
    return sqrt(mean((reference - prediction)**2))