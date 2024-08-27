from pathlib import Path

import numpy as np
import pytest

from msl.nlf import GaussianModel, load
from msl.nlf.loader import _load


def test_pi(tmp_path: Path) -> None:
    area, mu, sigma = (83.12, -1.3, 0.94)
    x = np.linspace(-10, 10)
    y = area / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    path = tmp_path / "gaussian.nlf"

    with GaussianModel(normalized=True) as model:
        assert model.equation == "a1/(a3*(2*pi)^0.5)*exp(-0.5*((x-a2)/a3)^2)"

        result = model.fit(x, y, params=(100, 1, 1))
        assert pytest.approx(result.params["a1"].value, abs=1e-12) == area
        assert pytest.approx(result.params["a2"].value, abs=1e-12) == mu
        assert pytest.approx(result.params["a3"].value, abs=1e-12) == sigma
        for a in ("a1", "a2", "a3"):
            assert result.params[a].uncert < 1e-14

        y_fit = model.evaluate(x, result)
        assert np.sum(np.abs(y_fit - y)) < 1e-15

        model.save(path)

    private = _load(path)
    assert private["equation"] == f"a1/(a3*(2*{np.pi})^0.5)*exp(-0.5*((x-a2)/a3)^2)"

    loaded = load(path)
    assert loaded.equation == "a1/(a3*(2*pi)^0.5)*exp(-0.5*((x-a2)/a3)^2)"
