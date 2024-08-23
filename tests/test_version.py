import sys

import pytest

from msl.nlf import LinearModel


def test_default() -> None:
    with LinearModel() as m:
        version = m.version()
        assert version == "5.44"
        assert m.version() is version  # value gets cached


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_win32() -> None:
    with LinearModel(win32=True) as m:
        version = m.version()
        assert version == "5.44"
        assert m.version() is version  # value gets cached
