import sys

import pytest

from msl.nlf import LinearModel


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_win32_windows() -> None:
    with LinearModel(win32=True) as m:
        assert m.delphi_library.name == "nlf-windows-i386.dll"

    name = "nlf-windows-x86_64.dll" if sys.maxsize > 2**32 else "nlf-windows-i386.dll"
    with LinearModel(win32=False) as m:
        assert m.delphi_library.name == name


@pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
def test_win32_unix() -> None:
    with pytest.raises(ValueError, match="'win32' feature is only supported on Windows"):
        _ = LinearModel(win32=True)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_win32_preserved_binary_operator() -> None:
    # The original NLF library path that was chosen is preserved when a binary operator is applied
    first_model_name = "nlf-windows-x86_64.dll" if sys.maxsize > 2**32 else "nlf-windows-i386.dll"

    m = LinearModel(win32=True) + 1
    assert str(m) == "Model(equation='a1+a2*x+1')"
    assert m.delphi_library.name == "nlf-windows-i386.dll"

    m2 = LinearModel(win32=True) + LinearModel()
    assert str(m2) == "CompositeModel(equation='(a1+a2*x)+(a3*x)')"
    assert m2.delphi_library.name == "nlf-windows-i386.dll"

    m3 = LinearModel() - LinearModel(win32=True)
    assert str(m3) == "CompositeModel(equation='(a1+a2*x)-(a3*x)')"
    assert m3.delphi_library.name == first_model_name

    m4 = LinearModel(win32=True) * LinearModel()
    assert str(m4) == "CompositeModel(equation='(a1+a2*x)*(a3+a4*x)')"
    assert m4.delphi_library.name == "nlf-windows-i386.dll"

    m5 = LinearModel() / LinearModel(win32=True)
    assert str(m5) == "CompositeModel(equation='(a1+a2*x)/(a3+a4*x)')"
    assert m5.delphi_library.name == first_model_name
