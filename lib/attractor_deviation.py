
from typing import Callable
from numba import njit
from numpy import absolute, array, empty, load, log2, maximum, minimum, ndarray, ndindex, zeros, zeros_like

from lib.attractor_inclusion import AttractorInclusion

class AttractorDeviation:
    def __init__(self, fname: str, normalizer: Callable[[ndarray], ndarray] | None = None, resolution : int | None  = None) -> None:
        """Initilizes the AttractorInclusion by loading the corresponding file. It uses the scale of the Lorenz system can be transform to any scale by using a Normalizer.

        It calculates how many points of the timeseries are inside the reference dataset.

        The scaling works by deviding the grid by 2**scaling, and thus, total_grid_sizes of two are important to use this feature.
        :param str fname: _description_
        :param UniformNormalizer | None normalier: _description_, defaults to None
        """
        data = load(fname)
        self.data = (data["hist"] > 0).astype("int")
        self.orig_n_buf = data["n_buf"]
        self.orig_n_bins = data["n_bins"]
        self.ma = data["maximum"]
        self.mi = data["minimum"]

        if resolution is None:
            scaling = 0
        else:
            if resolution > len(self.data):
                raise Exception("Upscaling is not supported")
            else:
                scaling = int(log2(len(self.data) // resolution))

        self.factor = scaling

        if self.orig_n_bins + 2*self.orig_n_buf != self.data.shape[-1]:
            raise Exception(f"The data and its name is not the same {self.orig_n_bins} vs {self.data.shape[-1]}")

        if scaling != 0:
            if scaling not in list(range(1, int(log2(self.orig_n_bins+2*self.orig_n_buf))+1)):
                raise Exception(f"Factor not in range, {range(1, int(log2(self.orig_n_bins+2*self.orig_n_buf)))}?" )
            self.data = AttractorInclusion.scale_down(self.data, scaling)

        if normalizer is not None:
            self.ma = normalizer(self.ma).squeeze()
            self.mi = normalizer(self.mi).squeeze()

        self.data = self.data.astype(int)

        print(f"Resolution is set to be: {self.data.shape}")


    @staticmethod
    @njit
    def _set_coordinates_to_histogram(coor: ndarray, histogram: ndarray):
        """Add each coordinate entry to the histogram

        :param ndarray coor: Array of coordinates
        :param ndarray adev: histogram
        """
        for x, y, z in coor:
            histogram[x,y,z] = 1

    def __call__(self, timeseries: ndarray, normalizer: Callable[[ndarray], ndarray] | None = None) -> float:
        """Evaluates the Attractor Inclusion

        :param ndarray timeseries: The timeseries for which you want the AInc value
        :param Callable[[ndarray], ndarray] | None normalizer: A function that is used to normalizes a datapoint, defaults to None
        :return float: AInc value
        """
        if normalizer is not None:
            ma = normalizer(self.ma).squeeze()
            mi = normalizer(self.mi).squeeze()
        else:
            ma, mi = self.ma, self.mi

        coor = AttractorInclusion._coordinates(timeseries, mi, ma, self.orig_n_buf, self.orig_n_bins)//2**self.factor
        histogram = zeros_like(self.data)
        self._set_coordinates_to_histogram(coor, histogram)
        return (absolute(histogram - self.data)).sum()
  