import numpy as np


class ZData:
    """
    A class for storing the sample of z-points.

    Parameters
    __________
    zre : ndarray
        Array containing the real parts.
    zim : ndarray
        Array containing the imaginary parts.
    env_shape : int
        The dimension of the z-sample.
    z : ndarray
        The z-points.
    z_conj : ndarray
        The complex conjugates of the z-points.

    """
    def __init__(self, randomize=False, size=180):
        if randomize:
            self.zre = np.random.uniform(0.5, 0.8, size=size)
            self.zim = np.random.uniform(0, 0.8, size=size)
            self.zim[self.zim == 0] += 1e-8
        else:
            self.zre = np.array([0.5414285714285715, 0.6357142857142857, 0.5728571428571428,
                             0.5257142857142857, 0.51, 0.5571428571428572, 0.73,
                             0.5257142857142857, 0.6985714285714286, 0.5257142857142857,
                             0.6514285714285715, 0.6828571428571428, 0.62, 0.6828571428571428,
                             0.6042857142857143, 0.7142857142857143, 0.6671428571428571,
                             0.6357142857142857, 0.6514285714285715, 0.5885714285714285,
                             0.5728571428571428, 0.6357142857142857, 0.5571428571428572,
                             0.5728571428571428, 0.5414285714285715, 0.5728571428571428, 0.62,
                             0.62, 0.5257142857142857, 0.5885714285714285, 0.62,
                             0.6514285714285715, 0.5571428571428572, 0.5728571428571428,
                             0.5414285714285715, 0.6828571428571428, 0.6357142857142857, 0.62,
                             0.6985714285714286, 0.5728571428571428, 0.5885714285714285, 0.51,
                             0.6985714285714286, 0.6514285714285715, 0.5885714285714285,
                             0.5728571428571428, 0.5885714285714285, 0.5414285714285715,
                             0.5257142857142857, 0.62, 0.6042857142857143, 0.6042857142857143,
                             0.6985714285714286, 0.5571428571428572, 0.6357142857142857,
                             0.5414285714285715, 0.73, 0.5257142857142857, 0.6514285714285715,
                             0.51, 0.62, 0.5885714285714285, 0.5414285714285715,
                             0.7142857142857143, 0.6357142857142857, 0.7142857142857143, 0.51,
                             0.51, 0.5885714285714285, 0.5414285714285715, 0.62,
                             0.6985714285714286, 0.6042857142857143, 0.6514285714285715,
                             0.5885714285714285, 0.6828571428571428, 0.73, 0.6671428571428571,
                             0.5728571428571428, 0.6828571428571428, 0.62, 0.5257142857142857,
                             0.5257142857142857, 0.5571428571428572, 0.5414285714285715,
                             0.6357142857142857, 0.5257142857142857, 0.62, 0.51,
                             0.6671428571428571, 0.6671428571428571, 0.5885714285714285,
                             0.6671428571428571, 0.6671428571428571, 0.5414285714285715,
                             0.6828571428571428, 0.5885714285714285, 0.51, 0.5414285714285715,
                             0.6671428571428571, 0.5728571428571428, 0.6671428571428571,
                             0.5257142857142857, 0.6042857142857143, 0.5257142857142857,
                             0.5728571428571428, 0.73, 0.5885714285714285, 0.6357142857142857,
                             0.51, 0.51, 0.6042857142857143, 0.73, 0.7142857142857143,
                             0.5257142857142857, 0.5571428571428572, 0.62, 0.5571428571428572,
                             0.5728571428571428, 0.5571428571428572, 0.6042857142857143,
                             0.7142857142857143, 0.6514285714285715, 0.6514285714285715,
                             0.5885714285714285, 0.51, 0.51, 0.6042857142857143,
                             0.6042857142857143, 0.6828571428571428, 0.51, 0.5257142857142857,
                             0.5571428571428572, 0.73, 0.7142857142857143, 0.6042857142857143,
                             0.5257142857142857, 0.73, 0.6985714285714286, 0.6828571428571428,
                             0.6671428571428571, 0.6514285714285715, 0.6357142857142857, 0.73,
                             0.5571428571428572, 0.5885714285714285, 0.7142857142857143,
                             0.6514285714285715, 0.5414285714285715, 0.5571428571428572,
                             0.5414285714285715, 0.5571428571428572, 0.5414285714285715,
                             0.5571428571428572, 0.6357142857142857, 0.5257142857142857, 0.51,
                             0.5414285714285715, 0.51, 0.6985714285714286, 0.5571428571428572,
                             0.6357142857142857, 0.6357142857142857, 0.62, 0.6514285714285715,
                             0.5728571428571428, 0.51, 0.5414285714285715, 0.7142857142857143,
                             0.6042857142857143, 0.6042857142857143, 0.5728571428571428,
                             0.6985714285714286, 0.5728571428571428, 0.6828571428571428,
                             0.6042857142857143, 0.62, 0.5728571428571428, 0.5571428571428572,
                             0.6514285714285715])
            self.zim = np.array([0.6071428571428571, 0.4414285714285714, 0.4414285714285714,
                             0.23428571428571426, 0.39999999999999997, 0.11, 0.2757142857142857,
                             0.6071428571428571, 0.5657142857142857, 0.69, 0.5657142857142857,
                             0.5657142857142857, 0.23428571428571426, 0.4828571428571428,
                             0.3171428571428571, 0.39999999999999997, 0.3171428571428571,
                             0.1514285714285714, 0.4414285714285714, 0.4414285714285714,
                             0.19285714285714284, 0.39999999999999997, 0.5657142857142857,
                             0.5657142857142857, 0.4828571428571428, 0.11, 0.5657142857142857,
                             0.3171428571428571, 0.19285714285714284, 0.19285714285714284,
                             0.4828571428571428, 0.35857142857142854, 0.35857142857142854,
                             0.4828571428571428, 0.11, 0.39999999999999997, 0.6071428571428571,
                             0.5242857142857142, 0.4828571428571428, 0.5242857142857142,
                             0.23428571428571426, 0.6071428571428571, 0.4414285714285714,
                             0.2757142857142857, 0.39999999999999997, 0.3171428571428571,
                             0.5657142857142857, 0.6485714285714285, 0.1514285714285714,
                             0.4414285714285714, 0.39999999999999997, 0.5242857142857142,
                             0.23428571428571426, 0.39999999999999997, 0.3171428571428571,
                             0.5657142857142857, 0.35857142857142854, 0.5657142857142857,
                             0.19285714285714284, 0.6485714285714285, 0.39999999999999997,
                             0.4828571428571428, 0.4414285714285714, 0.35857142857142854,
                             0.5242857142857142, 0.3171428571428571, 0.1514285714285714,
                             0.5242857142857142, 0.5242857142857142, 0.1514285714285714,
                             0.19285714285714284, 0.35857142857142854, 0.6485714285714285,
                             0.4828571428571428, 0.1514285714285714, 0.23428571428571426,
                             0.39999999999999997, 0.4414285714285714, 0.6485714285714285,
                             0.4414285714285714, 0.6485714285714285, 0.11, 0.2757142857142857,
                             0.2757142857142857, 0.39999999999999997, 0.19285714285714284,
                             0.5242857142857142, 0.1514285714285714, 0.4414285714285714,
                             0.2757142857142857, 0.5657142857142857, 0.3171428571428571,
                             0.23428571428571426, 0.5242857142857142, 0.19285714285714284,
                             0.3171428571428571, 0.6071428571428571, 0.23428571428571426,
                             0.3171428571428571, 0.39999999999999997, 0.35857142857142854,
                             0.35857142857142854, 0.4828571428571428, 0.35857142857142854,
                             0.6485714285714285, 0.23428571428571426, 0.4414285714285714,
                             0.6485714285714285, 0.2757142857142857, 0.11, 0.69,
                             0.2757142857142857, 0.23428571428571426, 0.11, 0.4414285714285714,
                             0.6485714285714285, 0.6071428571428571, 0.6071428571428571,
                             0.39999999999999997, 0.69, 0.4828571428571428, 0.4828571428571428,
                             0.23428571428571426, 0.3171428571428571, 0.35857142857142854,
                             0.5657142857142857, 0.19285714285714284, 0.4414285714285714,
                             0.6071428571428571, 0.2757142857142857, 0.35857142857142854,
                             0.39999999999999997, 0.3171428571428571, 0.3171428571428571,
                             0.2757142857142857, 0.5657142857142857, 0.3171428571428571,
                             0.5242857142857142, 0.39999999999999997, 0.5242857142857142,
                             0.4828571428571428, 0.5242857142857142, 0.4828571428571428,
                             0.4828571428571428, 0.4828571428571428, 0.2757142857142857,
                             0.4414285714285714, 0.6071428571428571, 0.5242857142857142,
                             0.23428571428571426, 0.23428571428571426, 0.1514285714285714, 0.69,
                             0.5242857142857142, 0.5657142857142857, 0.35857142857142854,
                             0.2757142857142857, 0.35857142857142854, 0.4828571428571428,
                             0.3171428571428571, 0.19285714285714284, 0.35857142857142854,
                             0.23428571428571426, 0.2757142857142857, 0.39999999999999997,
                             0.1514285714285714, 0.3171428571428571, 0.2757142857142857,
                             0.5242857142857142, 0.19285714285714284, 0.1514285714285714, 0.69,
                             0.5242857142857142, 0.2757142857142857, 0.35857142857142854,
                             0.23428571428571426, 0.35857142857142854, 0.6071428571428571,
                             0.4414285714285714, 0.1514285714285714])
        self.env_shape = self.zre.size
        self.z = self.zre + self.zim * 1j
        self.z_conj = self.z.conjugate()

    def kill_data(self, kill_list):
        """
        Deletes a number of z-points, their complex conjugates and recalculates the dimension.

        Parameters
        ----------
        kill_list : list
            A list of z-point positions to remove.
        """
        self.z = np.delete(self.z, kill_list)
        self.z_conj = np.delete(self.z_conj, kill_list)
        self.env_shape = self.z.size
