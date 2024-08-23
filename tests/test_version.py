import pytest

from msl.nlf import LinearModel
from msl.nlf.model import IS_PYTHON_64BIT


def test_default() -> None:
    with LinearModel() as m:
        version = m.version()
        assert version == "5.44"
        assert m.version() is version  # value gets cached


def test_nlf32() -> None:
    with LinearModel(dll="nlf32") as m:
        assert m.version() == "5.44"


@pytest.mark.skipif(not IS_PYTHON_64BIT, reason="32-bit Python cannot load 64-bit DLL")
def test_nlf64() -> None:
    with LinearModel(dll="nlf64") as m:
        assert m.version() == "5.44"
