import pytest

from msl.nlf import ConstantModel, ExponentialModel, GaussianModel, LinearModel, Model, PolynomialModel, SineModel
from msl.nlf.delphi import NPAR, NVAR


@pytest.mark.parametrize(
    "equation",
    [
        "a1*x+a2*x1+a3*x2",  # cannot use both x and x1
        "a1*exp(a2*x1+x/a3)",  # cannot use both x and x1
        "x0",  # cannot use x0
        "a1*x0",  # cannot use x0
        "x0*a1+2",  # cannot use x0
        "a1*exp(-a2*x0)",  # cannot use x0
        "a1*exp(-x0*a2)",  # cannot use x0
        "a0",  # cannot use a0
        "a0/x-273.15",  # cannot use a0
        "a1*(1-x)+x*a0",  # cannot use a0
        "+".join(f"x{i}" for i in range(1, NVAR + 2)),  # too many x's
        "+".join(f"a{i}*x" for i in range(1, NPAR + 2)),  # too many a's
    ],
)
def test_invalid(equation: str) -> None:
    # Test for potential invalid characters or too many variables/parameters.
    # Does not test whether the equation is a valid mathematical expression,
    # for example, missing a closing bracket or misspelling cos as coz
    with pytest.raises(ValueError, match="equation"):
        Model(equation)


@pytest.mark.parametrize(
    ("equation", "np", "nv"),
    [
        ("", 0, 0),
        ("a1", 1, 0),
        ("-a1", 1, 0),
        ("x", 0, 1),
        ("-x", 0, 1),
        ("a1*x", 1, 1),
        ("x*a1", 1, 1),
        ("a1*x1", 1, 1),
        ("x1*a1", 1, 1),
        ("a1+a2*(x+exp(a3*x))+x2", 3, 2),
        ("a1+a2*(x1+exp(a3*x1))+x2", 3, 2),
        ("a1*x+a2*sin(a4*x-a3)+x*a5*exp(-0.5((x-a6)/a7)^2+a8", 8, 1),
        ("x1*tan(a1)-a2*arcsin(a3-x2)/(a4+a6*arccos(a5-x2))+a7", 7, 2),
        ("a1*x1+3.2-a3*x2+log(3.2)/(a2*x3)+a4*x4-45.623", 4, 4),
        ("x4*exp(x1-x2/exp(x3))-exp(x5)+x6", 0, 6),
    ],
)
def test_npar_nvar(equation: str, np: int, nv: int) -> None:
    m = Model(equation)
    assert m.num_parameters == np
    assert m.num_variables == nv
    assert m.equation == equation
    assert m.user_function_name == ""


def test_property() -> None:
    m = Model("a1*x")
    assert m.equation == "a1*x"
    with pytest.raises(AttributeError):
        m.equation = "a1*x+a2"  # type: ignore[misc]


def test_linear() -> None:
    with LinearModel() as m:
        assert m.equation == "a1+a2*x"


def test_sine() -> None:
    with SineModel() as m:
        assert m.equation == "a1*sin(2*pi*a2*x+a3)"

    with SineModel(angular=True) as m:
        assert m.equation == "a1*sin(a2*x+a3)"


def test_gaussian() -> None:
    with GaussianModel(normalized=True) as m:
        assert m.equation == "a1/(a3*(2*pi)^0.5)*exp(-0.5*((x-a2)/a3)^2)"

    with GaussianModel() as m:
        assert m.equation == "a1*exp(-0.5*((x-a2)/a3)^2)"


def test_exponential() -> None:
    with ExponentialModel() as m:
        assert m.equation == "a1*exp(-a2*x)"


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "a1+a2*x"),
        (2, "a1+a2*x+a3*x^2"),
        (3, "a1+a2*x+a3*x^2+a4*x^3"),
        (4, "a1+a2*x+a3*x^2+a4*x^3+a5*x^4"),
        (5, "a1+a2*x+a3*x^2+a4*x^3+a5*x^4+a6*x^5"),
        (6, "a1+a2*x+a3*x^2+a4*x^3+a5*x^4+a6*x^5+a7*x^6"),
        (7, "a1+a2*x+a3*x^2+a4*x^3+a5*x^4+a6*x^5+a7*x^6+a8*x^7"),
        (12, "a1+a2*x+a3*x^2+a4*x^3+a5*x^4+a6*x^5+a7*x^6+a8*x^7+a9*x^8+a10*x^9+a11*x^10+a12*x^11+a13*x^12"),
    ],
)
def test_polynomial(n: int, expected: str) -> None:
    with PolynomialModel(n) as m:
        assert m.equation == expected


def test_gaussian_constant() -> None:
    m = GaussianModel(normalized=True) + ConstantModel()
    assert m.equation == "(a1/(a3*(2*pi)^0.5)*exp(-0.5*((x-a2)/a3)^2))+(a4)"

    m = ConstantModel() - GaussianModel(normalized=True)
    assert m.equation == "(a1)-(a2/(a4*(2*pi)^0.5)*exp(-0.5*((x-a3)/a4)^2))"

    m = GaussianModel() + ConstantModel()
    assert m.equation == "(a1*exp(-0.5*((x-a2)/a3)^2))+(a4)"

    m = ConstantModel() - GaussianModel()
    assert m.equation == "(a1)-(a2*exp(-0.5*((x-a3)/a4)^2))"


def test_gaussian_numeric_add() -> None:
    m = GaussianModel() + 1
    assert m.equation == "a1*exp(-0.5*((x-a2)/a3)^2)+1"

    m = 1 + GaussianModel()
    assert m.equation == "1+a1*exp(-0.5*((x-a2)/a3)^2)"

    m = GaussianModel() + 1.2
    assert m.equation == "a1*exp(-0.5*((x-a2)/a3)^2)+1.2"

    m = 1.2 + GaussianModel()
    assert m.equation == "1.2+a1*exp(-0.5*((x-a2)/a3)^2)"

    with pytest.raises(TypeError, match=r"unsupported operand"):
        GaussianModel() + 2j  # type: ignore[operator]

    with pytest.raises(TypeError, match=r"unsupported operand"):
        2j + GaussianModel()  # type: ignore[operator]


def test_gaussian_numeric_sub() -> None:
    m = GaussianModel() - 1
    assert m.equation == "a1*exp(-0.5*((x-a2)/a3)^2)-1"

    m = 1 - GaussianModel()
    assert m.equation == "1-a1*exp(-0.5*((x-a2)/a3)^2)"

    m = GaussianModel() - 1.2
    assert m.equation == "a1*exp(-0.5*((x-a2)/a3)^2)-1.2"

    m = 1.2 - GaussianModel()
    assert m.equation == "1.2-a1*exp(-0.5*((x-a2)/a3)^2)"

    with pytest.raises(TypeError, match=r"unsupported operand"):
        GaussianModel() - 2j  # type: ignore[operator]

    with pytest.raises(TypeError, match=r"unsupported operand"):
        2j - GaussianModel()  # type: ignore[operator]


def test_gaussian_numeric_mul() -> None:
    m = GaussianModel() * 1
    assert m.equation == "(a1*exp(-0.5*((x-a2)/a3)^2))*1"

    m = 1 * GaussianModel()
    assert m.equation == "1*(a1*exp(-0.5*((x-a2)/a3)^2))"

    m = GaussianModel() * 1.2
    assert m.equation == "(a1*exp(-0.5*((x-a2)/a3)^2))*1.2"

    m = 1.2 * GaussianModel()
    assert m.equation == "1.2*(a1*exp(-0.5*((x-a2)/a3)^2))"

    with pytest.raises(TypeError, match=r"unsupported operand"):
        GaussianModel() * 2j  # type: ignore[operator]

    with pytest.raises(TypeError, match=r"unsupported operand"):
        2j * GaussianModel()  # type: ignore[operator]


def test_gaussian_numeric_truediv() -> None:
    m = GaussianModel() / 1
    assert m.equation == "(a1*exp(-0.5*((x-a2)/a3)^2))/1"

    m = 1 / GaussianModel()
    assert m.equation == "1/(a1*exp(-0.5*((x-a2)/a3)^2))"

    m = GaussianModel() / 1.2
    assert m.equation == "(a1*exp(-0.5*((x-a2)/a3)^2))/1.2"

    m = 1.2 / GaussianModel()
    assert m.equation == "1.2/(a1*exp(-0.5*((x-a2)/a3)^2))"

    with pytest.raises(TypeError, match=r"unsupported operand"):
        GaussianModel() / 2j  # type: ignore[operator]

    with pytest.raises(TypeError, match=r"unsupported operand"):
        2j / GaussianModel()  # type: ignore[operator]


def test_sine_constant() -> None:
    m = SineModel(angular=True) + ConstantModel()
    assert m.equation == "(a1*sin(a2*x+a3))+(a4)"

    m = ConstantModel() - SineModel()
    assert m.equation == "(a1)-(a2*sin(2*pi*a3*x+a4))"


def test_exponential_constant() -> None:
    m = ExponentialModel() + ConstantModel()
    assert m.equation == "(a1*exp(-a2*x))+(a3)"

    m = ConstantModel() - ExponentialModel(cumulative=True)
    assert m.equation == "(a1)-(a2*(1-exp(-a3*x)))"


def test_exponential_sine() -> None:
    m = ExponentialModel() * SineModel()
    assert m.equation == "(a1*exp(-a2*x))*(sin(2*pi*a3*x+a4))"

    m = SineModel(angular=True) / ExponentialModel(cumulative=True)
    assert m.equation == "(a1*sin(a2*x+a3))/((1-exp(-a4*x)))"

    m = ExponentialModel() + SineModel(angular=True)
    assert m.equation == "(a1*exp(-a2*x))+(a3*sin(a4*x+a5))"

    m = SineModel() - ExponentialModel()
    assert m.equation == "(a1*sin(2*pi*a2*x+a3))-(a4*exp(-a5*x))"


def test_exponential_gaussian() -> None:
    m = ExponentialModel() / GaussianModel()
    assert m.equation == "(a1*exp(-a2*x))/(exp(-0.5*((x-a3)/a4)^2))"

    m = ExponentialModel(cumulative=True) / GaussianModel(normalized=True)
    assert m.equation == "(a1*(1-exp(-a2*x)))/(1/a4*exp(-0.5*((x-a3)/a4)^2))"


def test_gaussian_linear() -> None:
    m = GaussianModel(normalized=True) + LinearModel()
    assert m.equation == "(a1/(a3*(2*pi)^0.5)*exp(-0.5*((x-a2)/a3)^2))+(a4+a5*x)"

    m = LinearModel() - GaussianModel()
    assert m.equation == "(a1+a2*x)-(a3*exp(-0.5*((x-a4)/a5)^2))"

    m = GaussianModel() * LinearModel()
    assert m.equation == "(a1*exp(-0.5*((x-a2)/a3)^2))*(a4+a5*x)"

    m = LinearModel() / GaussianModel(normalized=True)
    assert m.equation == "(a1+a2*x)/(a3/(a5*(2*pi)^0.5)*exp(-0.5*((x-a4)/a5)^2))"

    m = LinearModel() / (LinearModel() * GaussianModel())
    assert m.equation == "(a1+a2*x)/((a3+a4*x)*(a5*exp(-0.5*((x-a6)/a7)^2)))"


def test_polynomial_polynomial() -> None:
    m = PolynomialModel(6) + PolynomialModel(5)
    assert m.equation == "(a1+a2*x+a3*x^2+a4*x^3+a5*x^4+a6*x^5+a7*x^6)+(a8*x+a9*x^2+a10*x^3+a11*x^4+a12*x^5)"


def test_polynomial_custom() -> None:
    m = PolynomialModel(12) + Model("a1 * x + a4 * arcsin( a3 * x - a2 )")
    assert m.equation == (
        "(a1+a2*x+a3*x^2+a4*x^3+a5*x^4+a6*x^5+a7*x^6+a8*x^7+a9*x^8+a10*x^9+a11*x^10+a12*x^11+a13*x^12)"
        "+"
        "(a14 * x + a17 * arcsin( a16 * x - a15 ))"
    )


def test_polynomial_constant() -> None:
    m = PolynomialModel(1) - Model("")
    assert m.equation == "a1+a2*x"

    # not using a predefined Model, so a3 still exists
    m = PolynomialModel(1) + Model("a1")
    assert m.equation == "(a1+a2*x)+(a3)"

    m = PolynomialModel(2) - ConstantModel()
    assert m.equation == "a1+a2*x+a3*x^2"

    m = PolynomialModel(2) * ConstantModel()
    assert m.equation == "(a1+a2*x+a3*x^2)*(a4)"


def test_sine_constant_exponential() -> None:
    m = ExponentialModel() * (SineModel() + ConstantModel())
    assert m.equation == "(a1*exp(-a2*x))*((a3*sin(2*pi*a4*x+a5))+(a6))"


def test_linear_linear() -> None:
    m = LinearModel() * LinearModel()
    assert m.equation == "(a1+a2*x)*(a3+a4*x)"


def test_numeric_exponential_linear_polynomial() -> None:
    m = (1 - ExponentialModel()) * LinearModel() / (2.1345 * PolynomialModel(2) + 273.15)
    assert m.equation == "((1-a1*exp(-a2*x))*(a3+a4*x))/(2.1345*(a5+a6*x+a7*x^2)+273.15)"


def test_delphi_raises() -> None:
    # arctan is not supported
    x = [-4868.68, -4868.09, -4867.41, -3375.19, -3373.14, -3372.03]
    y = [0.252429, 0.252141, 0.251809, 0.297989, 0.296257, 0.295319]
    with Model("a1 - a2*x - arctan(a3/(x-a4))/2.0") as model:
        assert model.num_variables == 1
        assert model.num_parameters == 4
        with pytest.raises(RuntimeError, match="Invalid Equation"):
            model.fit(x, y, params=[0.1, -1e-5, 1e3, -1e2])
