from typing import Any
from numpy import absolute, array, empty, linalg, mean, median, ndarray, quantile, sqrt

from models.lib.model import Model

class LocalStepAheadPredictability:
    @staticmethod
    def _calc_euclidian_distances(timeseries: ndarray, true_model: Model, window_lengths: list[int] = [1,]) -> list[ndarray]:
        """Calculates the eulidican norm between a window_length - ahead prediction of the reservoir vs the true model for all datapoints in the timeseries.

        :param ndarray timeseries: ndarray n_timesteps x n_problem_dimension        
        :param Model true_model: true Model to compare with.
        :param list[int] window_lengths: a list of different window length that should be testes, defaults to [1,]
        :return list[ndarray]: a list of the results for the different window lengths.
        """
        out = [empty((len(timeseries) - w)) for w in window_lengths]
        w_max = max(window_lengths)
        
        for i, state in enumerate((timeseries)):
            lorenz_max_prediction = true_model.evolve_system(state, w_max+1)
            for j, wj in enumerate(window_lengths):
                if i < len(timeseries) - wj:
                    out[j][i] = linalg.norm(lorenz_max_prediction[wj] - timeseries[i+wj])

        return out
    
    def __call__(self, timeseries: ndarray, true_model: Model, window_lengths: list[int] = [1,]) -> ndarray:
        """Calculates the eulidican norm between a window_length - ahead prediction of the reservoir vs the true model for all datapoints in the timeseries.

        :param ndarray timeseries: ndarray n_timesteps x n_problem_dimension        
        :param Model true_model: true Model to compare with.
        :param list[int] window_lengths: a list of different window length that should be testes, defaults to [1,]
        :return list[ndarray]: a array of the means
        """
        return array(
            [
                mean(distances) for distances in self._calc_euclidian_distances(timeseries, true_model, window_lengths)
            ]
        )