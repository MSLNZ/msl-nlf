import pytest

from msl.nlf import LinearModel
from msl.nlf import PolynomialModel
from msl.nlf.model import CompositeModel


@pytest.mark.parametrize(
    'op', ['-*', '*-', '/*', '/+', 'x', '\\', '-+', '<', '*+',
           '//', '>', '+-', '@', '+/', '+*', '*/', '/-', '**',
           '^', '--', '++', '-/', '+=', '-=', '*=', '/='])
def test_invalid_op(op):
    m1 = LinearModel()
    m2 = LinearModel()
    with pytest.raises(ValueError, match=r'Unsupported operator'):
        CompositeModel(op, m1, m2)


@pytest.mark.parametrize('n', [-99, -1.5, 0, 0.99999])
def test_polynomial_order_invalid(n):
    with pytest.raises(ValueError, match=r'must be >= 1'):
        PolynomialModel(n)
