from math import fsum
from math import sqrt

from pytest import approx

from msl.nlf import Model


def test_weighted_mean():
    y = [4.731, 10.624, 9.208, 6.178, 7.65, 10.133, 9.487,
         9.932, 3.74, 8.531, 7.97, 10.275]

    uy = [0.933, 0.951, 0.883, 1.246, 1.504, 1.042, 0.791,
          1.06, 0.831, 0.881, 1.311, 0.727]

    params = [1]

    weights = [1.0/(u*u) for u in uy]
    mean = fsum(w_i * y_i for w_i, y_i in zip(weights, y)) / fsum(weights)
    uncert = 1.0 / sqrt(fsum(weights))
    chisq = fsum(w_i * (y_i - mean)**2 for w_i, y_i in zip(weights, y))
    eof = sqrt(fsum((y_i - mean)**2 for y_i in y) / float(len(y) - len(params)))

    with Model('a1', weighted=True) as model:
        result = model.fit(range(len(y)), y, uy=uy, params=params)
        assert approx(mean, rel=1e-10) == result.params['a1'].value
        assert approx(uncert, rel=1e-10) == result.params['a1'].uncert
        assert approx(chisq, rel=1e-10) == result.chisq
        assert approx(eof, rel=1e-10) == result.eof
