import numpy as np
import pytest

from msl.nlf import LinearModel, Model, PolynomialModel
from msl.nlf.delphi import NPAR, NPTS, NVAR


def test_x_linear_model() -> None:
    model = LinearModel()

    # too many x variables, this error is raised by numpy
    with pytest.raises(ValueError, match="could not broadcast"):
        model.fit(x=np.empty((NVAR + 1, 5)), y=[], params=[])

    # too many points, this error is raised by numpy
    with pytest.raises(ValueError, match="could not broadcast"):
        model.fit(x=np.empty(NPTS + 1), y=[], params=[])

    # dimension too high
    with pytest.raises(ValueError, match=r"An array of shape \(5, 6, 7\)"):
        model.fit(x=np.empty((5, 6, 7)), y=[], params=[])

    # dimension too high
    with pytest.raises(ValueError, match=r"An array of shape \(2, 2, 2\)"):
        model.fit(x=[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], y=[], params=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"An array of shape \(1, 1, 2\)"):
        model.fit(x=[[[1.2, 3.4]]], y=[], params=[])  # type: ignore[arg-type]

    # expect a 1D array
    with pytest.raises(ValueError, match=r"Unexpected number of x \(stimulus\) variables"):
        model.fit(x=[[1, 2], [3, 4]], y=[], params=[])

    # this is ok
    model.fit(x=[1, 2], y=[1, 2], params=[1, 2], debug=True)
    model.fit(x=np.array([1, 2]), y=[1, 2], params=[1, 2], debug=True)


def test_x_model() -> None:
    model = Model("x1+a1*x2-x3+x4/a2")
    with pytest.raises(ValueError, match=r"Unexpected number of x \(stimulus\) variables"):
        model.fit(x=[1, 2], y=[], params=[])
    with pytest.raises(ValueError, match=r"Unexpected number of x \(stimulus\) variables"):
        model.fit(x=[[1, 2], [3, 4]], y=[], params=[])
    with pytest.raises(ValueError, match=r"Unexpected number of x \(stimulus\) variables"):
        model.fit(x=[[1, 2], [3, 4], [5, 6]], y=[], params=[])
    with pytest.raises(ValueError, match=r"Unexpected number of x \(stimulus\) variables"):
        model.fit(x=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], y=[], params=[])

    # this is ok
    model.fit(x=[[1, 2], [3, 4], [5, 6], [7, 8]], y=[1, 2], params=[1, 2], debug=True)


def test_y() -> None:
    model = LinearModel()

    # these error is raised by numpy
    with pytest.raises(IndexError, match=r"array is 1-dimensional"):
        model.fit(x=[], y=[[1, 2], [3, 4]], params=[])  # type: ignore[list-item]
    with pytest.raises(IndexError, match=r"array is 1-dimensional"):
        model.fit(x=[], y=[[1, 2, 3, 4]], params=[])  # type: ignore[list-item]
    with pytest.raises(IndexError, match=r"array is 1-dimensional"):
        model.fit(x=[], y=((1, 2), (3, 4)), params=[])  # type: ignore[arg-type]
    with pytest.raises(IndexError, match=r"array is 1-dimensional"):
        model.fit(x=[], y=((1, 2, 3, 4),), params=[])  # type: ignore[arg-type]

    # too many points, this error is raised by numpy
    with pytest.raises(ValueError, match="could not broadcast"):
        model.fit(x=[], y=np.empty(NPTS + 1), params=[])

    # dimension too high
    with pytest.raises(ValueError, match=r"An array of shape \(2, 3, 4\)"):
        model.fit(x=[], y=np.empty((2, 3, 4)), params=[])

    # not the same length as x
    with pytest.raises(ValueError, match=r"len\(y\) != len\(x\)"):
        model.fit(x=[1, 2, 3, 4], y=[1, 2], params=[])

    # these are ok
    model.fit(x=[1, 2], y=[1, 2], params=[1, 2], debug=True)
    model.fit(x=[1, 2], y=np.array([1, 2]), params=[1, 2], debug=True)


def test_param_linear_model() -> None:
    model = LinearModel()

    # too few parameters
    with pytest.raises(ValueError, match=r"Unexpected number of parameters"):
        model.fit(x=[1, 2], y=[1, 2], params=[])
    with pytest.raises(ValueError, match=r"Unexpected number of parameters"):
        model.fit(x=[1, 2], y=[1, 2], params=[1])

    # too many parameters
    with pytest.raises(ValueError, match=r"Unexpected number of parameters"):
        model.fit(x=[1, 2], y=[1, 2], params=[1, 2, 3])

    # this is ok
    model.fit(x=[1, 2], y=[1, 2], params=[1, 2], debug=True)

    # too few parameters
    params = model.create_parameters([("a1", 1)])
    with pytest.raises(ValueError, match=r"Unexpected number of parameters"):
        model.fit(x=[1, 2], y=[1, 2], params=params)

    # too many parameters
    params = model.create_parameters([("a1", 1), ("a2", 2), ("a3", 3)])
    with pytest.raises(ValueError, match=r"Unexpected number of parameters"):
        model.fit(x=[1, 2], y=[1, 2], params=params)

    # this is ok
    params = model.create_parameters([("a1", 1), ("a2", 2)])
    model.fit(x=[1, 2], y=[1, 2], params=params, debug=True)


def test_param_poly_model() -> None:
    model = PolynomialModel(7)

    # too few parameters
    with pytest.raises(ValueError, match=r"Unexpected number of parameters"):
        model.fit(x=[1, 2], y=[1, 2], params=[1, 2, 3])

    # too many parameters
    params = model.create_parameters()
    for i in range(1, 10):
        params[f"a{i}"] = i
    with pytest.raises(ValueError, match=r"Unexpected number of parameters"):
        model.fit(x=[1, 2], y=[1, 2], params=params)

    # this is ok
    params = model.create_parameters()
    for i in [1, 2, 3, 4, 5, 6, 7, 8]:
        params[f"a{i}"] = i
    model.fit(x=[1, 2], y=[1, 2], params=params, debug=True)
    model.fit(x=[1, 2], y=[1, 2], params=[1, 2, 3, 4, 5, 6, 7, 8], debug=True)

    # too many parameters
    with pytest.raises(ValueError, match="Invalid parameter name"):
        model.fit(x=[], y=[], params=np.empty(NPAR + 1))


def test_ux() -> None:
    model = LinearModel()

    with pytest.raises(ValueError, match=r"x.shape != ux.shape"):
        model.fit(x=[1, 2], y=[1, 2], params=[1, 2], ux=[1, 2, 3])

    with pytest.raises(ValueError, match=r"x.shape != ux.shape"):
        model.fit(x=[1, 2], y=[1, 2], params=[1, 2], ux=[[1, 2]])

    # this error is raised by numpy
    with pytest.raises(ValueError, match=r"could not broadcast"):
        model.fit(x=[1, 2], y=[1, 2], params=[1, 2], ux=np.empty((NVAR + 1, 3)))

    # this error is raised by numpy
    with pytest.raises(ValueError, match=r"could not broadcast"):
        model.fit(x=[1, 2], y=[1, 2], params=[1, 2], ux=np.empty(NPTS + 1))

    with pytest.raises(ValueError, match=r"An array of shape \(2, 2, 2\)"):
        model.fit(x=[1, 2], y=[1, 2], params=[1, 2], ux=[[[0, 0], [0, 0]], [[0, 0], [0, 0]]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"An array of shape \(1, 1, 2\)"):
        model.fit(x=[1, 2], y=[1, 2], params=[1, 2], ux=[[[1, 2]]])  # type: ignore[arg-type]

    # x.shape=(1, 2) with ux.shape=(2,) is okay
    model.fit(x=[[1, 2]], y=[1, 2], params=[1, 1], ux=[0.1, 0.2], weighted=True, debug=True)
    # x.shape=(1, 2) with ux.shape=(3,) is not okay
    with pytest.raises(ValueError, match=r"x.shape != ux.shape"):
        model.fit(x=[[1, 2]], y=[1, 2], params=[1, 1], ux=[0.1, 0.2, 0.3])


def test_uy() -> None:
    model = LinearModel()

    with pytest.raises(ValueError, match=r"len\(y\) != len\(uy\)"):
        model.fit(x=[1, 2], y=[1, 2], params=[1, 2], uy=[1, 2, 3])

    # this error is raised by numpy
    with pytest.raises(IndexError, match=r"array is 1-dimensional"):
        model.fit(x=[1, 2], y=[1, 2], params=[1, 2], uy=[[1, 2]])  # type: ignore[list-item]
    with pytest.raises(IndexError, match=r"array is 1-dimensional"):
        model.fit(x=[1, 2], y=[1, 2], params=[1, 2], uy=((1, 2),))  # type: ignore[arg-type]

    # this error is raised by numpy
    with pytest.raises(ValueError, match=r"could not broadcast"):
        model.fit(x=[1, 2], y=[1, 2], params=[1, 2], uy=np.empty(NPTS + 1))
