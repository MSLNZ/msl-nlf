from pathlib import Path

import numpy as np
import pytest
from numpy.typing import ArrayLike

from msl.nlf import LinearModel, Model


def save_corr_array(tmp_path: Path, array: ArrayLike, name: str) -> Path:
    f = tmp_path / f"CorrCoeffs {name}.txt"
    np.savetxt(f, array)
    return f


def test_set_correlation_dir_raises() -> None:
    # the Model type is irrelevant
    with LinearModel() as model:
        with pytest.raises(OSError, match=r"not a valid directory"):
            model.set_correlation_dir("does-not-exist")

        # empty string valid
        model.set_correlation_dir("")

        # must still be a string, even when call with None
        model.set_correlation_dir(None)
        assert model._corr_dir == ""  # noqa: SLF001


def test_set_correlation_dir(tmp_path: Path) -> None:  # noqa: PLR0915
    dummy1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    dummy2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    dummy3 = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    kwargs = {"x": [], "y": [], "params": [1, 2], "debug": True}

    # the Model type is irrelevant
    with LinearModel() as model:
        model.show_warnings = False

        # no correlation files exist
        model.set_correlation_dir(Path(__file__).parent)
        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 0
            assert np.array_equal(c.is_correlated, [[False, False], [False, False]])

        model.set_correlation_dir(tmp_path)

        # The correlations are independent of whether fit(correlated=False) or fit(correlated=True)
        #
        # The correlated flag is used by the NLF library to decide if correlations are to be used

        # create Y-Y correlation file
        y_y_file = save_corr_array(tmp_path, dummy1, "Y-Y")
        assert y_y_file.exists()
        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 1
            assert c.data[0].path == str(y_y_file)
            assert np.array_equal(c.data[0].coefficients, dummy1)
            assert np.array_equal(c.is_correlated, [[True, False], [False, False]])

        y_y_file.unlink()
        assert not y_y_file.exists()

        # create Y-X1 correlation file
        y_x1_file = save_corr_array(tmp_path, dummy2, "Y-X1")
        assert y_x1_file.exists()
        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 1
            assert c.data[0].path == str(y_x1_file)
            assert np.array_equal(c.data[0].coefficients, dummy2)
            assert np.array_equal(c.is_correlated, [[False, True], [True, False]])

        # create X1-X1 correlation file (Y-X1 still exists)
        x1_x1_file = save_corr_array(tmp_path, dummy3, "X1-X1")
        assert y_x1_file.exists()
        assert x1_x1_file.exists()
        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 2
            assert c.data[0].path == str(x1_x1_file)
            assert np.array_equal(c.data[0].coefficients, dummy3)
            assert c.data[1].path == str(y_x1_file)
            assert np.array_equal(c.data[1].coefficients, dummy2)
            assert np.array_equal(c.is_correlated, [[False, True], [True, True]])

        # create Y-Y correlation file (X1-X1 and Y-X1 still exist)
        y_y_file = save_corr_array(tmp_path, dummy1, "Y-Y")
        assert y_x1_file.exists()
        assert x1_x1_file.exists()
        assert y_y_file.exists()
        model.set_correlation_dir(tmp_path)
        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 3
            assert c.data[0].path == str(x1_x1_file)
            assert np.array_equal(c.data[0].coefficients, dummy3)
            assert c.data[1].path == str(y_x1_file)
            assert np.array_equal(c.data[1].coefficients, dummy2)
            assert c.data[2].path == str(y_y_file)
            assert np.array_equal(c.data[2].coefficients, dummy1)
            assert np.array_equal(c.is_correlated, [[True, True], [True, True]])

        # pretend that no correlation files exist
        model.set_correlation_dir(None)
        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 0
            assert np.array_equal(c.is_correlated, [[False, False], [False, False]])

        y_x1_file.unlink()
        x1_x1_file.unlink()
        y_y_file.unlink()


def test_set_correlation_raises() -> None:
    # the Model type is irrelevant
    with LinearModel() as model:
        for name in ["a1", " ", "y1"]:
            with pytest.raises(ValueError, match=r"Invalid correlation variable name"):
                model.set_correlation(name, "y", 1)

        for name in ["x0", "x3"]:
            with pytest.raises(ValueError, match=r"X index outside of range"):
                model.set_correlation("y", name, 1)

        for m in [[], [1, 2], np.empty((3, 4, 5))]:
            with pytest.raises(ValueError, match=r"Invalid correlation matrix dimension"):
                model.set_correlation("y", "y", m)

        with pytest.raises(ValueError, match=r"inhomogeneous shape"):
            model.set_correlation("y", "y", [[1, 2], [3]])


def test_set_correlation() -> None:  # noqa: PLR0915
    x = np.array([[1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]])
    y = np.array([1.1, 1.9, 3.2, 3.7])
    p = np.array([0.0, 0.9, 0.0])
    kwargs = {"x": x, "y": y, "params": p, "debug": True}

    # the Model type is irrelevant
    with Model("a1+a2*(x+exp(a3*x))+x2") as model:
        model.show_warnings = False

        # no correlations exist
        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 0
            assert np.array_equal(
                c.is_correlated, [[False, False, False], [False, False, False], [False, False, False]]
            )

        # The correlations are independent of whether fit(correlated=False) or fit(correlated=True)
        #
        # The correlated flag is used by the NLF library to decide if correlations are to be used

        # create Y-Y correlation file
        model.set_correlation("y", "y", 1)
        corr_dir = Path(model._corr_dir)  # noqa: SLF001

        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 1
            assert c.data[0].path == str(corr_dir / "CorrCoeffs Y-Y.txt")
            assert np.array_equal(c.data[0].coefficients, np.ones((4, 4)))
            assert np.array_equal(c.is_correlated, [[True, False, False], [False, False, False], [False, False, False]])

        model.remove_correlations()

        # create Y-X1 correlation file
        # 'x' gets automatically renamed to 'X1'
        # Y-Y correlations were removed
        y_x1_matrix = 7 * np.ones((4, 4))
        model.set_correlation("y", "x", y_x1_matrix)
        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 2
            assert c.data[0].path == str(corr_dir / "CorrCoeffs X1-Y.txt")
            assert c.data[1].path == str(corr_dir / "CorrCoeffs Y-X1.txt")
            assert np.array_equal(c.data[1].coefficients, y_x1_matrix)
            assert np.array_equal(c.data[0].coefficients, y_x1_matrix)
            assert np.array_equal(c.is_correlated, [[False, True, False], [True, False, False], [False, False, False]])

        # create bad X1-X1 correlation file, must have shape (4, 4)
        model.set_correlation("x", "x", np.ones((5, 5)))
        with pytest.raises(ValueError, match=r"Invalid 'X1-X1' correlation array shape"):
            model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]

        # create proper X1-X1 correlation file (Y-X1 still exists)
        x1_x1_matrix = 2 * np.ones((4, 4))
        model.set_correlation("x", "x", x1_x1_matrix)
        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 3
            assert c.data[0].path == str(corr_dir / "CorrCoeffs X1-X1.txt")
            assert c.data[1].path == str(corr_dir / "CorrCoeffs X1-Y.txt")
            assert c.data[2].path == str(corr_dir / "CorrCoeffs Y-X1.txt")
            assert np.array_equal(c.data[0].coefficients, x1_x1_matrix)
            assert np.array_equal(c.data[1].coefficients, y_x1_matrix)
            assert np.array_equal(c.data[2].coefficients, y_x1_matrix)
            assert np.array_equal(c.is_correlated, [[False, True, False], [True, True, False], [False, False, False]])

        # create Y-Y correlation file (X1-X1 and Y-X1 still exist)
        model.set_correlation("y", "y", 3)
        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 4
            assert c.data[0].path == str(corr_dir / "CorrCoeffs X1-X1.txt")
            assert c.data[1].path == str(corr_dir / "CorrCoeffs X1-Y.txt")
            assert c.data[2].path == str(corr_dir / "CorrCoeffs Y-X1.txt")
            assert c.data[3].path == str(corr_dir / "CorrCoeffs Y-Y.txt")
            assert np.array_equal(c.data[0].coefficients, x1_x1_matrix)
            assert np.array_equal(c.data[1].coefficients, y_x1_matrix)
            assert np.array_equal(c.data[2].coefficients, y_x1_matrix)
            assert np.array_equal(
                c.data[3].coefficients,
                np.array([[1.0, 3.0, 3.0, 3.0], [3.0, 1.0, 3.0, 3.0], [3.0, 3.0, 1.0, 3.0], [3.0, 3.0, 3.0, 1.0]]),
            )
            assert np.array_equal(c.is_correlated, [[True, True, False], [True, True, False], [False, False, False]])

        # create X2-Y correlation file (Y-Y, X1-X1 and Y-X1 still exist)
        model.set_correlation("x2", "y", 5)
        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 6
            assert c.data[0].path == str(corr_dir / "CorrCoeffs X1-X1.txt")
            assert c.data[1].path == str(corr_dir / "CorrCoeffs X1-Y.txt")
            assert c.data[2].path == str(corr_dir / "CorrCoeffs X2-Y.txt")
            assert c.data[3].path == str(corr_dir / "CorrCoeffs Y-X1.txt")
            assert c.data[4].path == str(corr_dir / "CorrCoeffs Y-X2.txt")
            assert c.data[5].path == str(corr_dir / "CorrCoeffs Y-Y.txt")
            assert np.array_equal(c.data[0].coefficients, x1_x1_matrix)
            assert np.array_equal(c.data[1].coefficients, y_x1_matrix)
            assert np.array_equal(
                c.data[2].coefficients,
                np.array([[1.0, 5.0, 5.0, 5.0], [5.0, 1.0, 5.0, 5.0], [5.0, 5.0, 1.0, 5.0], [5.0, 5.0, 5.0, 1.0]]),
            )
            assert np.array_equal(c.data[3].coefficients, y_x1_matrix)
            assert np.array_equal(
                c.data[4].coefficients,
                np.array([[1.0, 5.0, 5.0, 5.0], [5.0, 1.0, 5.0, 5.0], [5.0, 5.0, 1.0, 5.0], [5.0, 5.0, 5.0, 1.0]]),
            )
            assert np.array_equal(
                c.data[5].coefficients,
                np.array([[1.0, 3.0, 3.0, 3.0], [3.0, 1.0, 3.0, 3.0], [3.0, 3.0, 1.0, 3.0], [3.0, 3.0, 3.0, 1.0]]),
            )
            assert np.array_equal(c.is_correlated, [[True, True, True], [True, True, False], [True, False, False]])

        # again, no correlation files exist
        model.remove_correlations()
        for boolean in (False, True):
            _input = model.fit(correlated=boolean, **kwargs)  # type: ignore[call-overload]
            c = _input.correlations
            assert len(c.data) == 0
            assert np.array_equal(
                c.is_correlated, [[False, False, False], [False, False, False], [False, False, False]]
            )


def test_bad_corr_file(tmp_path: Path) -> None:
    # Delphi raises an error if the correlation file cannot be read

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    file = bad_dir / "CorrCoeffs Y-Y.txt"
    with file.open(mode="w") as fp:
        fp.write("1 2 3\n")
        fp.write("4 x 6\n")
        fp.write("7 8 9\n")

    with LinearModel(correlated=True, weighted=True) as model:
        model.set_correlation_dir(bad_dir)
        with pytest.raises(RuntimeError, match="Error reading the correlation coefficient file"):
            model.fit([1, 2, 3], [1, 2, 3], params=[0, 1], uy=[0.1, 0.1, 0.1])


def test_correlations_reset() -> None:
    # Prior to Delphi NLF version 5.43
    # (commit 457fcead50c9132a5599e38bf7e83a97f4e87cc9, 24 July 2023)
    # the correlation coefficients would only be read once (from disk)
    # and the same coefficients would be used subsequent correlated fits.
    #
    # This test checks that this issue does not occur.
    # The expected values were taken from the GUI for the same data

    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.1, 1.9, 3.2, 3.7]
    p = [0, 1]
    uy = [0.1, 0.2, 0.3, 0.4]

    with LinearModel(correlated=True) as model:
        model.show_warnings = False

        model.set_correlation("y", "y", 0.9)
        result = model.fit(x=x, y=y, params=p, uy=uy)
        assert pytest.approx(0.2, rel=1e-9) == result.params["a1"].value
        assert pytest.approx(0.0774596669241483, rel=1e-9) == result.params["a1"].uncert
        assert pytest.approx(0.91, rel=1e-9) == result.params["a2"].value
        assert pytest.approx(0.103247275993122, rel=1e-9) == result.params["a2"].uncert

        # the correlation is different so the uncertainties are different
        model.set_correlation("y", "y", 0.1)
        result = model.fit(x=x, y=y, params=p, uy=uy)
        assert pytest.approx(0.2, rel=1e-9) == result.params["a1"].value
        assert pytest.approx(0.232379000772445, rel=1e-9) == result.params["a1"].uncert
        assert pytest.approx(0.91, rel=1e-9) == result.params["a2"].value
        assert pytest.approx(0.126253712816693, rel=1e-9) == result.params["a2"].uncert
