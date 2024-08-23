from math import fsum, sqrt

import numpy as np
import pytest

from msl.nlf import Model


def test_weighted() -> None:
    y = [4.731, 10.624, 9.208, 6.178, 7.65, 10.133, 9.487, 9.932, 3.74, 8.531, 7.97, 10.275]
    uy = [0.933, 0.951, 0.883, 1.246, 1.504, 1.042, 0.791, 1.06, 0.831, 0.881, 1.311, 0.727]
    params = [1]
    x = np.ones(len(y))
    weights = [1.0 / (u * u) for u in uy]
    mean = fsum(w_i * y_i for w_i, y_i in zip(weights, y)) / fsum(weights)
    uncert = 1.0 / sqrt(fsum(weights))
    chisq = fsum(w_i * (y_i - mean) ** 2 for w_i, y_i in zip(weights, y))
    eof = sqrt(fsum((y_i - mean) ** 2 for y_i in y) / float(len(y) - len(params)))

    with Model("a1*x", weighted=True) as model:
        result = model.fit(x, y, uy=uy, params=params)
        assert pytest.approx(mean, rel=1e-10) == result.params["a1"].value
        assert pytest.approx(uncert, rel=1e-10) == result.params["a1"].uncert
        assert pytest.approx(chisq, rel=1e-10) == result.chisq
        assert pytest.approx(eof, rel=1e-10) == result.eof


def test_weighted_correlated() -> None:
    y = [4.731, 10.624, 9.208]
    uy = [0.933, 0.951, 0.883]
    x = np.ones(len(y))
    params = [1]

    correlation = np.array([[1.0, 0.15, 0.42], [0.15, 1.0, 0.86], [0.42, 0.86, 1.0]])

    covariance = np.array(
        [
            [uy[0] ** 2, correlation[0, 1] * uy[0] * uy[1], correlation[0, 2] * uy[0] * uy[2]],
            [correlation[0, 1] * uy[0] * uy[1], uy[1] ** 2, correlation[1, 2] * uy[1] * uy[2]],
            [correlation[0, 2] * uy[0] * uy[2], correlation[1, 2] * uy[1] * uy[2], uy[2] ** 2],
        ]
    )

    ones = np.ones(len(y))
    inv_covariance = np.linalg.inv(covariance)
    w = np.matmul(ones, inv_covariance)
    w /= np.matmul(w, ones)

    mean = fsum(w_i * y_i for w_i, y_i in zip(w, y))

    uncert = sqrt(
        w[0] ** 2 * uy[0] ** 2
        + w[1] ** 2 * uy[1] ** 2
        + w[2] ** 2 * uy[2] ** 2
        + 2 * correlation[0, 1] * w[0] * w[1] * uy[0] * uy[1]
        + 2 * correlation[0, 2] * w[0] * w[2] * uy[0] * uy[2]
        + 2 * correlation[1, 2] * w[1] * w[2] * uy[1] * uy[2]
    )

    chisq = 0.0
    for i in range(len(y)):
        for j in range(len(y)):
            chisq += inv_covariance[i, j] * (y[i] - mean) * (y[j] - mean)

    eof = sqrt(fsum((y_i - mean) ** 2 for y_i in y) / float(len(y) - len(params)))

    with Model("a1*x", weighted=True, correlated=True) as model:
        model.set_correlation("y", "y", matrix=correlation)
        result = model.fit(x, y, uy=uy, params=params)
        assert pytest.approx(mean, rel=1e-10) == result.params["a1"].value
        assert pytest.approx(uncert, rel=1e-10) == result.params["a1"].uncert
        assert pytest.approx(chisq, rel=1e-10) == result.chisq
        assert pytest.approx(eof, rel=1e-10) == result.eof


def test_weighted_correlated_linear(no_gtc: bool) -> None:  # noqa: FBT001
    x = [1.969, 4.981, 3.357]
    y = [4.731, 10.624, 9.208]
    uy = [0.933, 0.951, 0.883]
    params = [1, 1]
    correlation = np.array([[1.0, 0.15, 0.67], [0.15, 1.0, 0.39], [0.67, 0.39, 1.0]])

    covariance = np.array(
        [
            [uy[0] ** 2, correlation[0, 1] * uy[0] * uy[1], correlation[0, 2] * uy[0] * uy[2]],
            [correlation[0, 1] * uy[0] * uy[1], uy[1] ** 2, correlation[1, 2] * uy[1] * uy[2]],
            [correlation[0, 2] * uy[0] * uy[2], correlation[1, 2] * uy[1] * uy[2], uy[2] ** 2],
        ]
    )

    xt = np.vstack((np.ones(len(x)), x))
    inv_covariance = np.linalg.inv(covariance)
    cov = np.linalg.inv((xt @ inv_covariance) @ xt.T)
    a1, a2 = ((cov @ xt) @ inv_covariance) @ y
    ua1, ua2 = sqrt(cov[0, 0]), sqrt(cov[1, 1])
    a_corr = cov[0, 1] / (ua1 * ua2)

    chisq = 0.0
    for i in range(len(y)):
        for j in range(len(y)):
            chisq += inv_covariance[i, j] * (y[i] - (a1 + a2 * x[i])) * (y[j] - (a1 + a2 * x[j]))

    eof = sqrt(fsum((y[i] - (a1 + a2 * x[i])) ** 2 for i in range(len(y))) / float(len(y) - len(params)))

    with Model("a1+a2*x", weighted=True, correlated=True) as model:
        model.set_correlation("y", "y", matrix=correlation)
        result = model.fit(x, y, uy=uy, params=params)
        assert pytest.approx(a1, rel=1e-8) == result.params["a1"].value
        assert pytest.approx(ua1, rel=1e-10) == result.params["a1"].uncert
        assert pytest.approx(a2, rel=1e-10) == result.params["a2"].value
        assert pytest.approx(ua2, rel=1e-10) == result.params["a2"].uncert
        assert pytest.approx(chisq, rel=1e-10) == result.chisq
        assert pytest.approx(eof, rel=1e-10) == result.eof

        if no_gtc:
            pytest.skip("GTC cannot be imported, skipped at to_ureal() part")

        intercept, slope = result.to_ureal()
        assert intercept.get_correlation(intercept) == 1.0
        assert slope.get_correlation(slope) == 1.0
        assert pytest.approx(a_corr, rel=1e-10) == intercept.get_correlation(slope)
        assert pytest.approx(a_corr, rel=1e-10) == slope.get_correlation(intercept)
