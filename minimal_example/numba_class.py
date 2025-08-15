import numpy as np
import numba as nb


@nb.experimental.jitclass([("x", nb.float64[:, :])])
class Test:
    def __init__(self):
        self.x = np.arange(10000.0).reshape((10, 1000))

    def do_smth(self):
        rows, cols = self.x.shape
        sums = np.zeros(cols)
        for r in nb.prange(rows):
            print(r)
            sums[r] = np.sum(self.x[r])
        return np.sum(sums)

    def do_other(self):
        return do_other(self.x)


@nb.njit([nb.float64(nb.float64[:, :])], parallel=True)
def do_other(x):
    rows, cols = x.shape
    sums = np.zeros(cols)
    for r in nb.prange(rows):
        print(r)
        sums[r] = np.sum(x[r])
    return np.sum(sums)


t = Test()
print("Class member")
print(t.do_smth())
print("External function")
print(t.do_other())
