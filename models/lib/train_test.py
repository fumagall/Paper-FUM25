from numpy import ndarray, max, min

class UniformNormalizer:
    def __init__(self, list_of_timeseries: ndarray):
        """Provide a timeseries on which the normalization parameters are calculated.

        :param ndarray list_of_timeseries: A list of timeseries on which the parameters are calculated seprately.
        """
        self.ref_minimums = min(list_of_timeseries, axis=-2)[..., None, :]
        self.ref_differences = max(list_of_timeseries, axis=-2)[..., None, :] - self.ref_minimums

    def unnormalize(self, normalized_timeseries: ndarray) -> ndarray:
        return (normalized_timeseries * self.ref_differences) + self.ref_minimums

    def __call__(self, list_of_timeseries: ndarray) -> ndarray:
        """Normalizes the datasets by the calculated parameters from the init.

        :param ndarray list_of_timeseries: a list of timeseries
        :return ndarray: a list of the normalized timeseries
        """
        return (list_of_timeseries - self.ref_minimums) / self.ref_differences

def split_and_normalize_uniformly(list_of_timeseries: ndarray, n_test: int) -> tuple[ndarray, ndarray, UniformNormalizer]:
    """
    Splits the timeseries in a fromer part the train set and a later part the test set.
    :param timeseries: The timeseries that should be splitter
    :param n_test: Number of test samples
    :return: normalized train, normalized test, and a function to normalize.
    """
    train, test = list_of_timeseries[..., :-n_test, :], list_of_timeseries[..., -n_test:, :]
    normalizer = UniformNormalizer(train)
    return normalizer(train), normalizer(test), normalizer