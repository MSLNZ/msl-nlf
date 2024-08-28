import numpy as np
import pytest

from msl.nlf import ExponentialModel, LinearModel, Model, PolynomialModel, SineModel
from msl.nlf.datatypes import Result
from msl.nlf.parameters import InputParameters, ResultParameters


def test_sin() -> None:
    a1 = 3.4
    a2 = 10.3
    a3 = 0.6

    # pass in a name:value mapping
    x = np.linspace(0, 1, 100)
    with SineModel() as model:
        y = model.evaluate(x, {"a1": a1, "a2": a2, "a3": a3})
    assert np.allclose(a1 * np.sin(2 * np.pi * a2 * x + a3), y)

    result = Result(
        num_calls=1,
        chisq=1.0,
        correlation=np.zeros(10),
        covariance=np.zeros(10),
        dof=1,
        eof=1.0,
        iterations=1,
        params=ResultParameters(
            {"a": [a1, a2, a3], "ua": [0.1, 0.1, 0.1], "dof": 1}, InputParameters([("a1", a1), ("a2", a2), ("a3", a3)])
        ),
    )

    with SineModel() as model:
        y2 = model.evaluate(x, result)

    # pass in a Result object
    assert np.allclose(a1 * np.sin(2 * np.pi * a2 * x + a3), y2)


def test_cos() -> None:
    a1 = 6.1
    a2 = 9.5
    a3 = -1.4

    x = np.linspace(0, 1, 100)
    with Model("a1*cos(a2*x+a3)") as model:
        y = model.evaluate(x, {"a1": a1, "a2": a2, "a3": a3})

    assert np.allclose(a1 * np.cos(a2 * x + a3), y)


def test_linear() -> None:
    a1 = 0.2
    a2 = 9.5

    x = np.linspace(0, 1, 100)
    with LinearModel() as model:
        y = model.evaluate(x, {"a1": a1, "a2": a2})

    assert np.allclose(a1 + a2 * x, y)


def test_polynomial() -> None:
    a1 = 0.2
    a2 = 9.5
    a3 = 0.2
    a4 = 1e-2
    a5 = 2.1e-3
    a6 = 5.3e-4
    a7 = 8.7e-8

    x = np.linspace(0, 1, 100)
    with PolynomialModel(6) as model:
        y = model.evaluate(x, {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5, "a6": a6, "a7": a7})

    assert np.allclose(a1 + a2 * x + a3 * x**2 + a4 * x**3 + a5 * x**4 + a6 * x**5 + a7 * x**6, y)


def test_exponential() -> None:
    a1 = 11.1
    a2 = 2.3

    x = np.linspace(0, 1, 100)
    with ExponentialModel() as model:
        y = model.evaluate(x, {"a1": a1, "a2": a2})

    assert np.allclose(a1 * np.exp(-a2 * x), y)


def test_tan() -> None:
    a1 = 4.2
    a2 = 7.3
    a3 = -2.4

    x = np.linspace(0, 1, 100)
    with Model("a1*tan(a2*x+a3)") as model:
        y = model.evaluate(x, {"a1": a1, "a2": a2, "a3": a3})

    assert np.allclose(a1 * np.tan(a2 * x + a3), y)


def test_ln() -> None:
    a1 = 4.2
    a2 = 7.3

    x = np.linspace(1, 2, 100)
    with Model("a1+ln(x/a2)") as model:
        y = model.evaluate(x, {"a1": a1, "a2": a2})

    assert np.allclose(a1 + np.log(x / a2), y)


def test_log() -> None:
    a1 = 2.9
    a2 = 6.1

    x = np.linspace(1, 2, 100)
    with Model("a1+log(x/a2)") as model:
        y = model.evaluate(x, {"a1": a1, "a2": a2})

    assert np.allclose(a1 + np.log10(x / a2), y)


def test_arcsin() -> None:
    a1 = 4.2
    a2 = 0.97

    x = np.linspace(-1, 1, 100)
    with Model("a1*arcsin(a2*x)") as model:
        y = model.evaluate(x, {"a1": a1, "a2": a2})

    assert np.allclose(a1 * np.arcsin(a2 * x), y)


def test_arcos() -> None:
    a1 = 4.2
    a2 = 0.97

    x = np.linspace(-1, 1, 100)
    with Model("a1*arcos(a2*x)") as model:
        y = model.evaluate(x, {"a1": a1, "a2": a2})

    assert np.allclose(a1 * np.arccos(a2 * x), y)


def test_ugly() -> None:
    a1 = 4.2
    a2 = 0.97
    a3 = 8.6
    a4 = 0.7

    x = np.linspace(-1, 1, 100)
    equation = "(a1*arcos(a2*x)+273.15-a4*x^3)/(2*sin(a3*x)-8*exp(-a4*x)+ln(a3)*x)"
    with Model(equation) as model:
        y = model.evaluate(x, {"a1": a1, "a2": a2, "a3": a3, "a4": a4})

    assert np.allclose(
        (a1 * np.arccos(a2 * x) + 273.15 - a4 * x**3) / (2 * np.sin(a3 * x) - 8 * np.exp(-a4 * x) + np.log(a3) * x), y
    )


@pytest.mark.parametrize(
    "equation",
    [
        "a1+a2*(x+exp(a3*x))+x2",  # use x instead of x1
        "a1+a2*(x1+exp(a3*x1))+x2",
    ],
)  # use x1 and x2
def test_multiple_variables(equation: str) -> None:
    x = np.array([[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]])
    y = np.array([1.1, 1.9, 3.2, 3.7])
    guess = np.array([0, 0.9, 0])

    model = Model(equation)
    result = model.fit(x, y, params=guess)

    with pytest.raises(ValueError, match=r"Invalid shape of x data"):
        model.evaluate([[[1, 2], [3, 4], [5, 6]]], result)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"Unexpected number of x variables"):
        model.evaluate([1, 2, 3, 4], result)

    with pytest.raises(ValueError, match=r"Unexpected number of x variables"):
        model.evaluate([[1, 2], [3, 4], [5, 6]], result)

    y_fit = model.evaluate(x, result)
    assert pytest.approx(result.chisq) == np.sum((y - y_fit) ** 2)


@pytest.mark.parametrize(
    "equation",
    [
        "a1+a2*x",  # use x instead of x1
        "a1+a2*x1",
    ],
)  # use x1 instead of x
def test_1d_x_x1(equation: str) -> None:
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4])
    with Model(equation) as model:
        result = model.fit(x, y, params=[0, 1])
        y_fit = model.evaluate(x, result)
        assert pytest.approx(result.chisq) == np.sum((y - y_fit) ** 2)


@pytest.mark.parametrize(
    "code",
    [
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__build_class__",
        "__import__",
        "abs(-1)",
        "all([0, 1, 0])",
        'compile("import os", "<string>", "exec")',
        'compile("import sys", "<string>", "exec")',
        f"open({__file__!r})",
    ],
)
def test_builtins_removed(code: str) -> None:
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4])
    with LinearModel() as model:
        result = model.fit(x, y, params=[0, 1])
        model._equation = code  # noqa: SLF001
        with pytest.raises(NameError):
            model.evaluate(x, result)
