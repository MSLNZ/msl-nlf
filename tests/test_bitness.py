import pytest

from msl.nlf import LinearModel
from msl.nlf.dll import NPAR, NPTS, NVAR
from msl.nlf.model import IS_PYTHON_64BIT


def test_nlf32() -> None:
    # nlf32.dll can be loaded in both 32-bit and 64-bit Python
    expect = "LinearModel(equation='a1+a2*x', dll='nlf32.dll')"
    with LinearModel(dll="nlf32") as m:
        assert str(m) == expect
        assert repr(m) == expect
        assert str(m.dll_path) != "nlf32.dll"
        assert m.dll_path.name == "nlf32.dll"
        assert m.MAX_VARIABLES == NVAR
        assert m.MAX_POINTS == NPTS
        assert m.MAX_PARAMETERS == NPAR


def test_nlf64() -> None:
    dll = "nlf64"
    if IS_PYTHON_64BIT:
        expect = "LinearModel(equation='a1+a2*x', dll='nlf64.dll')"
        with LinearModel(dll=dll) as m:
            assert str(m) == expect
            assert repr(m) == expect
            assert str(m.dll_path) != "nlf64.dll"
            assert m.dll_path.name == "nlf64.dll"
    else:
        with pytest.raises(ValueError, match=r"64-bit DLL in 32-bit Python"):  # noqa: SIM117
            with LinearModel(dll=dll):
                pass


def test_dll_preserved_binary_operator() -> None:
    # the original DLL path that was chosen is preserved when a binary operator is applied
    m = LinearModel(dll="nlf32") + 1
    assert str(m) == "Model(equation='a1+a2*x+1', dll='nlf32.dll')"

    # gets the dll path from the left-hand side
    m2 = LinearModel(dll="nlf32") + LinearModel()
    assert str(m2) == "CompositeModel(equation='(a1+a2*x)+(a3*x)', dll='nlf32.dll')"
