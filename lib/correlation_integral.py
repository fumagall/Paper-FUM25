from abc import ABC, abstractmethod
from typing import Callable
from numpy import absolute, arange, array, fromiter, inf, linspace, load, log, ndarray, random, zeros, gradient
from numpy.fft import fft
from numpy.linalg import inv
from numpy.random import Generator, default_rng
from scipy.stats import chi2
from numba import njit

class TransformedCorrelationIntegral(ABC):
    def __init__(self, fname: str) -> None:
        d = load(fname)
        self.ref_timeseries = d["timeseries"]
        self.ref_mean = d["mean"]
        self.ref_cov_inv = inv(d["cov"])
        self.ref_cov = (d["cov"])
        self.rs = d["rs"]

    @staticmethod
    @abstractmethod
    def transform(timeseries: ndarray) -> ndarray:
        pass
    
    @staticmethod
    @abstractmethod
    def random_subset(rng, timeseries: ndarray) -> ndarray:
        pass

    @staticmethod
    @njit
    def correlation_sum_heikki(timeseries1: ndarray, timeseries2: ndarray, r_array: ndarray) -> ndarray:
        sum = zeros(r_array.shape, dtype="int")
        r_array = r_array**2
        devider = 0
        for x, y, z in timeseries1:
            for xj, yj, zj in timeseries2:
                dist = ((x-xj)**2 + (y-yj)**2 + (z-zj)**2)
                # if dist == 0:
                #     continue
                devider += 1
                for i in range(len(sum)):
                    if dist < r_array[i]:
                        sum[i] += 1
        return sum.astype("float") / devider
    
    @staticmethod
    @njit
    def distance_minmax(timeseries1: ndarray, timeseries2: ndarray) -> tuple[float, float]:
        min = inf
        max = -inf
        for x, y, z in timeseries1:
            for xj, yj, zj in timeseries2:
                dist = ((x-xj)**2 + (y-yj)**2 + (z-zj)**2)
                if dist == 0:
                    continue
                if dist < min:
                    min = dist
                if dist > max:
                    max = dist
        return min, max

    def __call__(self, timeseries, rng: Generator, unnormalizer: Callable[[ndarray], ndarray] | None = None):

        def worker(timeseries, r_timeseries):
            timeseries = self.random_subset(rng, timeseries)
            r_timeseries = self.random_subset(rng, r_timeseries)
            y =  self.correlation_sum_heikki(timeseries, r_timeseries, self.rs)
            return (y - self.ref_mean) @ self.ref_cov_inv @ (y - self.ref_mean).T       
        
        ref_timeseries = self.ref_timeseries
        if unnormalizer is not None:
            timeseries = unnormalizer(timeseries)

        timeseries = self.transform(timeseries)

        chi2_sum = fromiter(map(
            lambda r_timeseries: worker(timeseries, r_timeseries),
            ref_timeseries[:]
            ), float)

        chi2_sum.sort()
        squared_errors =  ((linspace(0, 1, len(chi2_sum)) - chi2.cdf(chi2_sum, df=len(self.ref_mean)))**2)
        test_statistic = squared_errors[len(squared_errors)//2] if len(squared_errors) % 2 == 1 else (squared_errors[len(squared_errors)//2] + squared_errors[len(squared_errors)//2-1])/2

        return test_statistic, chi2_sum

class SpatialCI(TransformedCorrelationIntegral):
    
    @staticmethod
    def transform(timeseries: ndarray) -> ndarray:
        return timeseries
    
    @staticmethod
    def random_subset(rng, timeseries: ndarray) -> ndarray:
        return timeseries[::3]
    
class SpatialCI02(TransformedCorrelationIntegral):
    
    @staticmethod
    def transform(timeseries: ndarray) -> ndarray:
        return timeseries
    
    @staticmethod
    def random_subset(rng, timeseries: ndarray) -> ndarray:
        return timeseries[::15]
    
class LongSpatialCI(TransformedCorrelationIntegral):
    @staticmethod
    def transform(timeseries: ndarray) -> ndarray:
        return timeseries
    
    @staticmethod
    def random_subset(rng, timeseries: ndarray) -> ndarray:
        return rng.choice(timeseries, len(timeseries)//3, replace=False)
    
    def __call__(self, timeseries, rng: Generator, unnormalizer: Callable[[ndarray], ndarray] | None = None) -> ndarray:
        _, chi2_sum = super().__call__(timeseries, rng, unnormalizer)
        return (-chi2.logsf(chi2_sum, df=len(self.ref_mean)), chi2_sum) # type: ignore

class LongSpatial02CI(TransformedCorrelationIntegral):
    @staticmethod
    def transform(timeseries: ndarray) -> ndarray:
        return timeseries
    
    @staticmethod
    def random_subset(rng, timeseries: ndarray) -> ndarray:
        return rng.choice(timeseries, len(timeseries)//15, replace=False) 
    
    def __call__(self, timeseries, rng: Generator, unnormalizer: Callable[[ndarray], ndarray] | None = None) -> ndarray:
        _, chi2_sum = super().__call__(timeseries, rng, unnormalizer)
        return (-chi2.logsf(chi2_sum, df=len(self.ref_mean)), chi2_sum) # type: ignore
    
    