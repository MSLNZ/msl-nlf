from __future__ import annotations

import pytest

from msl.nlf import LinearModel, PolynomialModel
from msl.nlf.model import CompositeModel


@pytest.mark.parametrize(
    "op",
    [
        "-*",
        "*-",
        "/*",
        "/+",
        "x",
        "\\",
        "-+",
        "<",
        "*+",
        "//",
        ">",
        "+-",
        "@",
        "+/",
        "+*",
        "*/",
        "/-",
        "**",
        "^",
        "--",
        "++",
        "-/",
        "+=",
        "-=",
        "*=",
        "/=",
    ],
)
def test_invalid_op(op: str) -> None:
    m1, m2 = LinearModel(), LinearModel()
    with pytest.raises(ValueError, match=r"Unsupported operator"):
        CompositeModel(op, m1, m2)


@pytest.mark.parametrize("n", [-99, -1.5, 0, 0.99999])
def test_polynomial_order_invalid(n: int | float) -> None:  # noqa: PYI041
    with pytest.raises(ValueError, match=r"must be >= 1"):
        PolynomialModel(n)  # type: ignore[arg-type]
