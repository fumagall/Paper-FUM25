from typing import Any, Callable

from numpy import array, floating, fromiter, linalg, load, log2, ndindex, prod, zeros, absolute
from numpy import ndarray
from scipy.stats import wasserstein_distance_nd

from .attractor_inclusion import AttractorInclusion


class TotalVariation:
    def __init__(self, fname: str, normalizer: Callable[[ndarray], ndarray] | None = None, resolution : int | None  = None) -> None:
        """Initilizes the Wasserstein measure by loading the corresponding file. It uses the scale of the Lorenz system can be transform to any scale by using a Normalizer.

        It compares the spacial distribution of a timeseries with the reference histogram using the WassersteinND distance.

        The scaling works by deviding the grid by 2**scaling, and thus, total_grid_sizes of two are important to use this feature.
        :param str fname: _description_
        :param UniformNormalizer | None normalier: _description_, defaults to None
        """
        data = load(fname)
        self.hist_ref = data["hist"]
        self.orig_n_buf = data["n_buf"]
        self.orig_n_bins = data["n_bins"]
        self.ma = data["maximum"]
        self.mi = data["minimum"]

        if resolution is None:
            scaling = 0
        else:
            if resolution > len(self.hist_ref):
                raise Exception("Upscaling is not supported")
            else:
                scaling = int(log2(len(self.hist_ref) // resolution))

        factor = scaling
        self.factor = scaling

        if self.orig_n_bins + 2*self.orig_n_buf != self.hist_ref.shape[-1]:
            if factor not in list(range(1, int(log2(self.orig_n_bins+2*self.orig_n_buf))+1)):
                raise Exception(f"Factor not in range, {range(1, int(log2(self.orig_n_bins+2*self.orig_n_buf)))}?" )
            raise Exception(f"The data and its name is not the same {self.orig_n_bins} vs {self.hist_ref.shape[-1]}")

        if factor != 0:
            self.hist_ref = AttractorInclusion.scale_down(self.hist_ref,  factor)

        self.hist_ref = self.hist_ref.astype(float) / self.hist_ref.sum()

        print(f"Resolution is set to be: {self.hist_ref.shape}")


        if normalizer is not None:
            self.ma = normalizer(self.ma).squeeze()
            self.mi = normalizer(self.mi).squeeze()

    def _calc_timeseries_histogram(self, timeseries: ndarray, normalizer: Callable[[ndarray], ndarray] | None = None) -> ndarray:
        if normalizer is not None:
            ma = normalizer(self.ma).squeeze()
            mi = normalizer(self.mi).squeeze()
        else:
            ma, mi = self.ma, self.mi

        hist_time = zeros(self.hist_ref.shape, dtype=int)
        coor = AttractorInclusion._coordinates(timeseries, mi, ma, self.orig_n_buf, self.orig_n_bins) // 2**self.factor
        AttractorInclusion._add_coordinates_to_histogram(coor, hist_time)
        hist_time = hist_time.astype(float) / hist_time.sum()
        return hist_time

    def __call__(self, timeseries: ndarray, normalizer: Callable[[ndarray], ndarray] | None = None) -> ndarray:
        """Calculates the frobenius norm of the difference of the histogram of the reference data and given timeseries as a histogram.

        :param ndarray timeseries: input
        :param Callable[[ndarray], ndarray] | None normalizer: a normalization function, defaults to None
        :return tuple[float, ndarray]: WassersteinND metric, Histogram of the timeseries 
        """
        diff = self._calc_timeseries_histogram(timeseries, normalizer) - self.hist_ref
        return 1/2*absolute(diff).sum()  # type: ignore
        