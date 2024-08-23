import sys

import pytest
from msl.loadlib import Server32Error  # type: ignore[import-untyped]

from msl.nlf import FitMethod, LinearModel, Model


def test_runtime_error() -> None:
    model = LinearModel(weighted=True)
    with pytest.raises(RuntimeError, match=r"The uncertainties are not complete"):
        model.fit(x=[1, 2, 3, 4], y=[1, 2, 3, 4], params=[1, 1])


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_server32_error() -> None:
    with LinearModel(weighted=True, win32=True) as model:  # noqa: SIM117
        with pytest.raises(Server32Error, match=r"The uncertainties are not complete"):
            model.fit(x=[1, 2, 3, 4], y=[1, 2, 3, 4], params=[1, 1])


@pytest.mark.parametrize("method", [FitMethod.AMOEBA_MD, FitMethod.POWELL_MD, FitMethod.AMOEBA_MM, FitMethod.POWELL_MM])
def test_correlated(method: FitMethod) -> None:
    # Minimum Distance and MiniMax are invalid for correlated data
    match = r"cannot be performed as a correlated fit"
    with LinearModel(fit_method=method, correlated=True) as model:
        model.set_correlation("x", "y", value=0.5)
        with pytest.raises(RuntimeError, match=match):
            model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[1, 1])


@pytest.mark.parametrize("method", [FitMethod.AMOEBA_MD, FitMethod.POWELL_MD])
def test_minimum_distance_nvars(method: FitMethod) -> None:
    # Minimum Distance is invalid if nvars > 1
    match = r"can only be performed for functions with one x-variable"
    model = Model("a1*x + a2*x2", fit_method=method)
    with pytest.raises(RuntimeError, match=match):
        model.fit(x=[[1, 2, 3], [1, 2, 3]], y=[1, 2, 3], params=[1, 1])
