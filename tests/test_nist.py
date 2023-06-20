import numpy as np
import pytest

from NIST_datasets._nist import NIST  # noqa
from msl.nlf import Model


def check_nist(
        dataset: str,
        rel: float = None,
        abs: float = None,  # noqa
        abs_chisqr: float = None,
        abs_eof: float = None,
        skip_guess1: bool = False,
        **options) -> None:
    """Assert that the NLF result is equivalent to the certified NIST result."""
    nist = NIST(dataset)
    with Model(nist.equation, **options) as model:
        for guess in (nist.guess1, nist.guess2):
            if skip_guess1 and guess is nist.guess1:
                continue
            y = np.log(nist.y) if nist.log_y else nist.y
            result = model.fit(nist.x, y, params=guess)
            assert nist.dof == result.dof
            assert pytest.approx(nist.chisqr, abs=abs_chisqr) == result.chisq
            assert pytest.approx(nist.eof, abs=abs_eof) == result.eof
            assert len(result.params) == len(nist.certified)
            for p in result.params:
                c = nist.certified[p.name]
                assert pytest.approx(p.value, rel=rel, abs=abs) == c['value']
                assert pytest.approx(p.uncert, rel=rel, abs=abs) == c['uncert']


def test_Misra1a():  # noqa
    check_nist('Misra1a')


def test_Chwirut2():  # noqa
    check_nist('Chwirut2')


def test_Chwirut1():  # noqa
    check_nist('Chwirut1')


def test_Lanczos3():  # noqa
    check_nist('Lanczos3', max_iterations=1500)


def test_Gauss1():  # noqa
    check_nist('Gauss1')


def test_Gauss2():  # noqa
    check_nist('Gauss2')


def test_DanWood():  # noqa
    check_nist('DanWood')


def test_Misra1b():  # noqa
    check_nist('Misra1b')


def test_Kirby2():  # noqa
    check_nist('Kirby2')


def test_Hahn1():  # noqa
    check_nist('Hahn1')


def test_Nelson():  # noqa
    check_nist('Nelson')


# def test_MGH17():  # noqa
#     check_nist('MGH17')


def test_Lanczos1():  # noqa
    check_nist('Lanczos1', max_iterations=1500)


def test_Lanczos2():  # noqa
    check_nist('Lanczos2', max_iterations=1500)


def test_Gauss3():  # noqa
    check_nist('Gauss3')


def test_Misra1c():  # noqa
    check_nist('Misra1c')


def test_Misra1d():  # noqa
    check_nist('Misra1d')


def test_Roszman1():  # noqa
    # Roszman1 requires arctan, which is not supported by the DLL
    nist = NIST('Roszman1')
    with Model(nist.equation) as model:
        with pytest.raises(RuntimeError, match='Invalid Equation'):
            model.fit(nist.x, nist.y, params=nist.guess1)


def test_ENSO():  # noqa
    check_nist('ENSO')


# def test_MGH09():  # noqa
#     check_nist('MGH09')


def test_Thurber():  # noqa
    check_nist('Thurber')


def test_BoxBOD():  # noqa
    # for guess1, get the following error
    #  RuntimeError: Error in Gauss-Jordan elimination routine - singular matrix.
    check_nist('BoxBOD', skip_guess1=True)


def test_Rat42():  # noqa
    check_nist('Rat42')


# def test_MGH10():  # noqa
#     check_nist('MGH10')


def test_Eckerle4():  # noqa
    check_nist('Eckerle4')


def test_Rat43():  # noqa
    check_nist('Rat43')


# def test_Bennett5():  # noqa
#     check_nist('Bennett5',
#                fit_method=Model.FitMethod.AMOEBA_LS,
#                rel=1e-4,
#                abs_chisqr=1e-8,
#                abs_eof=1e-8,
#                max_iterations=2000)
