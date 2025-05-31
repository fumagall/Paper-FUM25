
from typing import Callable
from numba import njit
from numpy import array, empty, load, log2, maximum, minimum, ndarray, ndindex, zeros

class AttractorInclusion:
    def __init__(self, fname: str, normalizer: Callable[[ndarray], ndarray] | None = None, resolution : int | None  = None) -> None:
        """Initilizes the AttractorInclusion by loading the corresponding file. It uses the scale of the Lorenz system can be transform to any scale by using a Normalizer.

        It calculates how many points of the timeseries are inside the reference dataset.

        The scaling works by deviding the grid by 2**scaling, and thus, total_grid_sizes of two are important to use this feature.
        :param str fname: _description_
        :param UniformNormalizer | None normalier: _description_, defaults to None
        """
        data = load(fname)
        self.hist = (data["hist"] > 0).astype("int")
        self.orig_n_buf = data["n_buf"]
        self.orig_n_bins = data["n_bins"]
        self.ma = data["maximum"]
        self.mi = data["minimum"]

        if resolution is None:
            scaling = 0
        else:
            if resolution > len(self.hist):
                raise Exception("Upscaling is not supported")
            else:
                scaling = int(log2(len(self.hist) // resolution))

        self.factor = scaling

        if self.orig_n_bins + 2*self.orig_n_buf != self.hist.shape[-1]:
            raise Exception(f"The data and its name is not the same {self.orig_n_bins} vs {self.hist.shape[-1]}")

        if scaling != 0:
            if scaling not in list(range(1, int(log2(self.orig_n_bins+2*self.orig_n_buf))+1)):
                raise Exception(f"Factor not in range, {range(1, int(log2(self.orig_n_bins+2*self.orig_n_buf)))}?" )
            self.hist = self.scale_down(self.hist, scaling)

        if normalizer is not None:
            self.ma = normalizer(self.ma).squeeze()
            self.mi = normalizer(self.mi).squeeze()

        print(f"Resolution is set to be: {self.hist.shape}")

    @staticmethod
    @njit
    def scale_down(data: ndarray, scaling: int) -> ndarray:
        """Scales down histogram

        :param ndarray data: histogram, dimensions must be of power of two
        :param int scaling: scaling factor, such that the histogram is scaled down by 2**sclaing
        :return ndarray: scaled down histogram
        """
        shape = array(data.shape, dtype="int")
        factor=2**scaling
        new_adev = zeros((shape[0]//factor, shape[1]//factor, shape[2]//factor))
        for xo, yo, zo in ndindex(new_adev.shape):
            for x, y, z in ndindex((factor, factor, factor)):
                new_adev[xo, yo, zo] = maximum(new_adev[xo, yo, zo], data[x+factor*xo, y+factor*yo, z+factor*zo])
        return new_adev

    @staticmethod
    @njit
    def _add_coordinates_to_histogram(coor: ndarray, histogram: ndarray):
        """Add each coordinate entry to the histogram

        :param ndarray coor: Array of coordinates
        :param ndarray adev: histogram
        """
        for x, y, z in coor:
            histogram[x,y,z] += 1

    @staticmethod
    @njit
    def _coordinates(timeseries: ndarray, mi: ndarray, ma: ndarray, n_buf: int, n_bins: int):
        """Calculates coordinates

        :param ndarray timeseries: Input
        :param ndarray mi: minimum entry
        :param ndarray ma: maximum entry
        :param int n_buf: buffer arround minimum and maximum entry
        :param int n_bins: number of regular bins 
        :return _type_: array of coordinates corresponding to the time samples
        """
        coor = ((timeseries - mi )*((n_bins-1)/(ma - mi))).astype("int") + n_buf
        coor = minimum(coor, 2*n_buf+n_bins -1)
        coor = maximum(coor, 0)
        return coor

    @classmethod
    def add_timeseries_to_histogram(cls, histogram: ndarray, timeseries: ndarray, mi: ndarray, ma: ndarray, n_buf:int, n_bins:int, normalizer: Callable[[ndarray], ndarray] | None = None):
        """Addes a timeseries by converting it to coordinates to the histogram

        :param ndarray histogram: Preallocates histogram
        :param ndarray timeseries: input array 
        :param ndarray mi: minimum entry
        :param ndarray ma: maximum entry
        :param int n_buf: buffer arround minimum and maximum entry
        :param int n_bins: number of bins inbetween minimum and maximum entry
        :param Callable[[ndarray], ndarray] | None normalizer: Function to normalize, defaults to None
        """
        coor = cls._coordinates(timeseries, mi, ma, n_buf, n_bins)
        cls._add_coordinates_to_histogram(coor, histogram)

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

        coor = self._coordinates(timeseries, mi, ma, self.orig_n_buf, self.orig_n_bins)//2**self.factor
        return self._measure(self.hist, coor)
        
    @staticmethod
    @njit
    def _measure(data: ndarray, coor: ndarray) -> float:
        """Measures AInc for given histogram and timeseries

        :param ndarray data: histogram
        :param ndarray coor: array of coordinates
        :return float: AInc value
        """
        sum = 0
        for x, y, z in coor:
            sum += float(data[x, y, z])
        return sum/len(coor)