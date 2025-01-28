from pathlib import Path

import numpy as np
import pytest

from msl.nlf import FitMethod, LinearModel, Model, ResidualType, load


def test_must_specify() -> None:
    with LinearModel() as model:
        with pytest.raises(ValueError, match=r"x data"):
            model.save("")
        with pytest.raises(ValueError, match=r"y data"):
            model.save("", x=[1, 2])
        with pytest.raises(ValueError, match=r"params"):
            model.save("", x=[1, 2], y=[1, 2])


def test_bad_extension() -> None:
    model = LinearModel()
    with pytest.raises(ValueError, match=r"\.nlf"):
        model.save("filename.msl", x=[1, 2], y=[3, 4], params=[1, 1])


def test_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "overwrite.nlf"
    with LinearModel() as model:
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[1, 1])
        model.save(path)
        with pytest.raises(FileExistsError):
            model.save(path)
        model.save(path, overwrite=True)


def test_default_x_y(tmp_path: Path) -> None:
    x = [[1000, 2000, 3000]]
    y = [0.1, 0.2, 0.3]
    path = tmp_path / "default_x_y.nlf"
    with LinearModel() as model:
        model.fit(x, y, params=[1, 1])
        model.save(path)

    loaded = load(path)
    assert loaded.num_variables == 1
    assert loaded.num_parameters == 2
    assert loaded.comments == ""
    assert loaded.equation == "a1+a2*x"
    assert loaded.nlf_path.endswith("x_y.nlf")
    assert loaded.nlf_version == "5.46"
    assert np.array_equal(loaded.x, x)
    assert np.array_equal(loaded.y, y)
    assert np.array_equal(loaded.ux, [[0, 0, 0]])
    assert np.array_equal(loaded.uy, [0, 0, 0])


def test_override_x_y(tmp_path: Path) -> None:
    path = tmp_path / "override_x_y.nlf"
    with LinearModel() as model:
        model.fit([1000, 2000, 3000], [0.1, 0.2, 0.3])

        model.save(path, x=[1, 2, 3])
        loaded = load(path)
        assert np.array_equal(loaded.x, [[1, 2, 3]])
        assert np.array_equal(loaded.y, [0.1, 0.2, 0.3])
        assert np.array_equal(loaded.ux, [[0, 0, 0]])
        assert np.array_equal(loaded.uy, [0, 0, 0])

        model.save(path, y=[-1, 2, -3], overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.x, [[1, 2, 3]])
        assert np.array_equal(loaded.y, [-1, 2, -3])
        assert np.array_equal(loaded.ux, [[0, 0, 0]])
        assert np.array_equal(loaded.uy, [0, 0, 0])

        model.save(path, y=[11, 22, 33], x=[[101, 202, 303]], overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.x, [[101, 202, 303]])
        assert np.array_equal(loaded.y, [11, 22, 33])
        assert np.array_equal(loaded.ux, [[0, 0, 0]])
        assert np.array_equal(loaded.uy, [0, 0, 0])

        model.fit([1000, 2000, 3000], [0.1, 0.2, 0.3])
        model.save(path, overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.x, [[1000, 2000, 3000]])
        assert np.array_equal(loaded.y, [0.1, 0.2, 0.3])
        assert np.array_equal(loaded.ux, [[0, 0, 0]])
        assert np.array_equal(loaded.uy, [0, 0, 0])


def test_override_params(tmp_path: Path) -> None:
    x = [1, 2, 3]
    y = [0.1, 0.2, 0.3]
    path = tmp_path / "override_params.nlf"
    with LinearModel() as model:
        # use guess
        model.fit(x, y)
        model.save(path)
        loaded = load(path)
        a, b = loaded.params.values()
        assert pytest.approx(a, abs=1e-15) == 0
        assert pytest.approx(b, abs=1e-15) == 0.1
        assert np.array_equal(loaded.params.constants(), [False, False])
        assert loaded.params.names() == ["a1", "a2"]
        assert loaded.params.labels() == [None, None]

        # define as array
        model.fit(x, y, params=[-1, 2])
        model.save(path, overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.params.values(), [-1, 2])
        assert np.array_equal(loaded.params.constants(), [False, False])
        assert loaded.params.names() == ["a1", "a2"]
        assert loaded.params.labels() == [None, None]

        # don't call fit before save
        model.save(path, params=[1, 2], overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.params.values(), [1, 2])
        assert np.array_equal(loaded.params.constants(), [False, False])
        assert loaded.params.names() == ["a1", "a2"]
        assert loaded.params.labels() == [None, None]

        # params not specified, so get the previous result
        model.save(path, overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.params.values(), [1, 2])
        assert np.array_equal(loaded.params.constants(), [False, False])
        assert loaded.params.names() == ["a1", "a2"]
        assert loaded.params.labels() == [None, None]

        # create InputParameters instance
        params = model.create_parameters(
            (
                ("a1", -5.5, True, "intercept"),
                ("a2", 10.1, False, "slope"),
            )
        )
        model.save(path, params=params, overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.params.values(), [-5.5, 10.1])
        assert np.array_equal(loaded.params.constants(), [True, False])
        assert loaded.params.names() == ["a1", "a2"]

        # since the Delphi GUI does not use labels, the labels
        # are not preserved during the save-load procedure
        assert loaded.params.labels() == [None, None]


def test_override_ux_uy(tmp_path: Path) -> None:
    x = [1, 2, 3]
    y = [0.1, 0.2, 0.3]
    path = tmp_path / "override_ux_uy.nlf"
    with LinearModel() as model:
        model.show_warnings = False  # suppress weighted fit warnings

        # all zeros
        model.fit(x, y)
        model.save(path)
        loaded = load(path)
        assert np.array_equal(loaded.x, [x])
        assert np.array_equal(loaded.y, y)
        assert np.array_equal(loaded.ux, [[0, 0, 0]])
        assert np.array_equal(loaded.uy, [0, 0, 0])

        # ux changes
        model.save(path, ux=[0.01, 0.02, 0.03], overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.x, [x])
        assert np.array_equal(loaded.y, y)
        assert np.array_equal(loaded.ux, [[0.01, 0.02, 0.03]])
        assert np.array_equal(loaded.uy, [0, 0, 0])

        # ux same as previous call, uy changes
        model.save(path, uy=[0.3, 0.1, 0.2], overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.x, [x])
        assert np.array_equal(loaded.y, y)
        assert np.array_equal(loaded.ux, [[0.01, 0.02, 0.03]])
        assert np.array_equal(loaded.uy, [0.3, 0.1, 0.2])


@pytest.mark.parametrize("comments", ["", "hello", "hello\nworld", "{hello}\n{world}", "the data=from today"])
def test_comments(tmp_path: Path, comments: str) -> None:
    path = tmp_path / "comments.nlf"
    with LinearModel() as model:
        model.fit([1, 2, 3], [0.1, 0.2, 0.3])
        model.save(path, comments=comments, overwrite=True)
        loaded = load(path)
        assert loaded.comments == comments


def test_save_demo(tmp_path: Path) -> None:  # noqa: PLR0915
    path = tmp_path / "demo.nlf"
    x = np.array([[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]])
    y = np.array([1.1, 1.9, 3.2, 3.7])
    a = np.array([0, 0.9, 0])
    uy = np.array([0.5, 0.5, 0.5, 0.5])
    ux = np.array([[0.01, 0.02, 0.03, 0.04], [0.002, 0.004, 0.006, 0.008]])
    with Model("a1+a2*(x+exp(a3*x))+x2", weighted=True, correlated=True) as model:
        model.set_correlation("y", "y", 0.5)
        model.set_correlation("x", "x", 0.8)
        model.options(second_derivs_B=False, max_iterations=200)
        result1 = model.fit(x, y, params=a, uy=uy, ux=ux, tolerance=1.23e-12, delta=0.01)
        model.save(path, comments="demo")

    loaded = load(path)
    assert loaded.num_variables == 2
    assert loaded.num_parameters == 3
    assert loaded.comments == "demo"
    assert loaded.equation == "a1+a2*(x+exp(a3*x))+x2"
    assert loaded.nlf_path.endswith("demo.nlf")
    assert loaded.nlf_version == "5.46"
    assert np.array_equal(loaded.x, x)
    assert np.array_equal(loaded.y, y)
    assert np.array_equal(loaded.ux, ux)
    assert np.array_equal(loaded.uy, uy)
    assert np.array_equal(loaded.params.values(), a)
    assert np.array_equal(loaded.params.constants(), [False, False, False])
    assert loaded.params.names() == ["a1", "a2", "a3"]
    assert loaded.params.labels() == [None, None, None]

    data = loaded.fit(loaded.x, loaded.y, params=loaded.params, ux=loaded.ux, uy=loaded.uy, debug=True)

    assert data.absolute_residuals is True
    assert data.correlated is True
    assert data.delta == 0.01
    assert data.equation == "a1+a2*(x+exp(a3*x))+x2"
    assert data.fit_method == FitMethod.LM
    assert data.residual_type == ResidualType.DY_X
    assert data.max_iterations == 200
    assert np.array_equal(data.params.values(), a)
    assert np.array_equal(data.params.constants(), [False, False, False])
    assert data.params.names() == ["a1", "a2", "a3"]
    assert data.params.labels() == [None, None, None]
    assert data.second_derivs_B is False
    assert data.second_derivs_H is True
    assert data.tolerance == 1.23e-12
    assert np.array_equal(data.ux, ux)
    assert np.array_equal(data.uy, uy)
    assert data.uy_weights_only is False
    assert data.weighted is True
    assert np.array_equal(data.x, x)
    assert np.array_equal(data.y, y)

    c = data.correlations
    assert np.array_equal(c.is_correlated, [[True, False, False], [False, True, False], [False, False, False]])
    assert c.data[0].path.endswith("X1-X1.txt")
    assert c.data[1].path.endswith("Y-Y.txt")
    assert np.array_equal(
        c.data[0].coefficients,
        np.array([[1.0, 0.8, 0.8, 0.8], [0.8, 1.0, 0.8, 0.8], [0.8, 0.8, 1.0, 0.8], [0.8, 0.8, 0.8, 1.0]]),
    )
    assert np.array_equal(
        c.data[1].coefficients,
        np.array([[1.0, 0.5, 0.5, 0.5], [0.5, 1.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.5], [0.5, 0.5, 0.5, 1.0]]),
    )

    result2 = loaded.fit(loaded.x, loaded.y, params=loaded.params, ux=loaded.ux, uy=loaded.uy)
    assert np.array_equal(result1.params.values(), result2.params.values())
    assert np.array_equal(result1.params.uncerts(), result2.params.uncerts())


def test_options(tmp_path: Path) -> None:
    path = tmp_path / "options.nlf"
    with LinearModel() as model:
        model.options(absolute_residuals=False)
        model.options(correlated=True)
        model.options(delta=3.2)
        model.options(max_iterations=9876)
        model.options(fit_method=FitMethod.POWELL_MM)
        model.options(residual_type=ResidualType.DY_Y)
        model.options(second_derivs_B=True)
        model.options(second_derivs_H=False)
        model.options(tolerance=5.4321e-6)
        model.options(uy_weights_only=True)
        model.options(weighted=True)
        model.save(path, x=[1, 2, 3], y=[1, 2, 3], params=[1, 1])
        loaded = load(path)
        assert loaded.equation == "a1+a2*x"
        loaded.show_warnings = False
        data = loaded.fit(loaded.x, loaded.y, params=loaded.params, debug=True)
        assert data.absolute_residuals is False
        assert data.correlated is True
        assert data.delta == 3.2
        assert data.max_iterations == 9876
        assert data.fit_method == FitMethod.POWELL_MM
        assert data.residual_type == ResidualType.DY_Y
        assert data.second_derivs_B is True
        assert data.second_derivs_H is False
        assert data.tolerance == 5.4321e-6
        assert data.uy_weights_only is True
        assert data.weighted is True
        assert data.equation == "a1+a2*x"
        assert data.params.names() == ["a1", "a2"]
        assert data.params.labels() == [None, None]
        assert np.array_equal(data.params.values(), [1.0, 1.0])
        assert np.array_equal(data.params.constants(), [False, False])
        assert np.array_equal(data.ux, [[0.0, 0.0, 0.0]])
        assert np.array_equal(data.uy, [0.0, 0.0, 0.0])
        assert np.array_equal(data.x, [[1.0, 2.0, 3.0]])
        assert np.array_equal(data.y, [1.0, 2.0, 3.0])
