import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import ArrayLike

from msl.nlf import LinearModel, Model


def save_corr_array(tmp_path: Path, array: ArrayLike, name: str) -> Path:
    f = tmp_path / f"CorrCoeffs {name}.txt"
    np.savetxt(f, array)
    return f


def test_fit_weighted() -> None:
    model = LinearModel()

    assert model.show_warnings is True

    # check with ux
    with pytest.warns(UserWarning, match="uncertainties are specified"):
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, weighted=False, ux=[0.1, 0.2, 0.3])

    # UserWarning is suppressed
    model.show_warnings = False
    assert model.show_warnings is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, weighted=False, ux=[0.1, 0.2, 0.3])

    # check with uy
    model.show_warnings = True
    assert model.show_warnings is True
    with pytest.warns(UserWarning, match="uncertainties are specified"):
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, weighted=False, uy=[0.1, 0.2, 0.3])


def test_fit_correlated(tmp_path: Path) -> None:
    model = LinearModel()

    assert model.show_warnings is True
    model.set_correlation("y", "y", 0.5)

    # check with set_correlation
    with pytest.warns(UserWarning, match="correlations are specified"):
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=False)

    # UserWarning is suppressed
    model.show_warnings = False
    assert model.show_warnings is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=False)

    # re-enable
    model.show_warnings = True
    with pytest.warns(UserWarning, match="correlations are specified"):
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=True)

    # remove
    model.remove_correlations()

    # no warnings
    assert model.show_warnings is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=False)

    # set directory
    save_corr_array(tmp_path, np.array([[1.0, 2.0], [3.0, 4.0]]), "Y-Y")
    model.set_correlation_dir(tmp_path)
    assert model.show_warnings is True
    with pytest.warns(UserWarning, match="correlations are specified"):
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=True)

    # remove
    model.remove_correlations()

    # no warnings
    assert model.show_warnings is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=False)

    # correlated but no correlations are specified
    with pytest.warns(UserWarning, match="no correlations"):
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=True)

    # disable
    model.show_warnings = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=True)


def test_max_iterations() -> None:
    x = np.array([[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]])
    y = np.array([1.1, 1.9, 3.2, 3.7])
    ux = np.array([[0.01, 0.02, 0.03, 0.04], [0.002, 0.004, 0.006, 0.008]])
    uy = np.array([0.5, 0.5, 0.5, 0.5])
    guess = np.array([0, 0.9, 0])
    model = Model("a1+a2*(x1+exp(a3*x1))+x2")
    model.options(weighted=True, correlated=True)
    model.set_correlation("y", "y", 0.5)
    model.set_correlation("x1", "x1", 0.8)

    assert model.show_warnings is True
    with pytest.warns(UserWarning, match="fit iterations exceeded"):
        model.fit(x, y, ux=ux, uy=uy, params=guess, max_iterations=30)

    # UserWarning is suppressed
    model.show_warnings = False
    assert model.show_warnings is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(x, y, ux=ux, uy=uy, params=guess, max_iterations=30)

    # re-enable
    model.show_warnings = True
    assert model.show_warnings is True
    with pytest.warns(UserWarning, match="fit iterations exceeded"):
        model.fit(x, y, ux=ux, uy=uy, params=guess, max_iterations=30)
