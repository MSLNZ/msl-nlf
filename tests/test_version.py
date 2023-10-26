import pytest

from msl.nlf import LinearModel
from msl.nlf import version_info
from msl.nlf.model import IS_PYTHON_64BIT


def test_default():
    with LinearModel() as m:
        version = m.version()
        assert version == f'{version_info.major}.{version_info.minor}'
        assert m.version() is version  # value gets cached


def test_nlf32():
    with LinearModel(dll='nlf32') as m:
        assert m.version() == f'{version_info.major}.{version_info.minor}'


@pytest.mark.skipif(not IS_PYTHON_64BIT, reason='32-bit Python cannot load 64-bit DLL')
def test_nlf64():
    with LinearModel(dll='nlf64') as m:
        assert m.version() == f'{version_info.major}.{version_info.minor}'
