import pytest
from msl.loadlib import IS_PYTHON_64BIT
from msl.loadlib import Server32Error

from msl.nlf import LinearModel


def test_runtime_error():
    with LinearModel(weighted=True) as model:
        with pytest.raises(RuntimeError, match=r'The uncertainties are not complete'):
            model.fit(x=[1, 2, 3, 4], y=[1, 2, 3, 4], params=[1, 1])


@pytest.mark.skipif(not IS_PYTHON_64BIT, reason='32-bit Python, expect RuntimeError not Server32Error')
def test_server32_error():
    with LinearModel(weighted=True, dll='nlf32') as model:
        with pytest.raises(Server32Error, match=r'The uncertainties are not complete'):
            model.fit(x=[1, 2, 3, 4], y=[1, 2, 3, 4], params=[1, 1])
