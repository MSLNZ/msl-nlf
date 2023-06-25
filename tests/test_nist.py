import pytest

from NIST_datasets._nist import NIST  # noqa
from msl.nlf import Model


def assert_nist(
        dataset: str,
        *,
        equation: str = None,
        relative: float = 1e-6,
        show_warnings: bool = True,
        **options) -> None:
    """Assert that the NLF result is equivalent to the NIST result.

    Parameters
    ----------
    dataset
        The name of a NIST dataset.
    equation
        The equation to use. Only used by Roszman1.
    relative
        The relative tolerance between the NLF result and the NIST result.
    show_warnings
        Whether warnings from the Model should be shown.
    **options
        All other keyword arguments are passed to the Model.
    """
    nist = NIST(dataset)
    with Model(equation or nist.equation, **options) as model:
        model.show_warnings = show_warnings
        for guess in (nist.guess1, nist.guess2):
            result = model.fit(nist.x, nist.y, params=guess)
            assert nist.dof == result.dof
            assert pytest.approx(nist.chisqr) == result.chisq
            assert pytest.approx(nist.eof) == result.eof
            assert len(result.params) == len(nist.certified)
            for p in result.params:
                c = nist.certified[p.name]
                assert pytest.approx(c['value'], rel=relative) == p.value
                assert pytest.approx(c['uncert'], rel=relative) == p.uncert


def test_Misra1a():  # noqa
    assert_nist('Misra1a')


def test_Chwirut2():  # noqa
    assert_nist('Chwirut2')


def test_Chwirut1():  # noqa
    assert_nist('Chwirut1')


def test_Lanczos3():  # noqa
    assert_nist('Lanczos3',
                max_iterations=1500)


def test_Gauss1():  # noqa
    assert_nist('Gauss1')


def test_Gauss2():  # noqa
    assert_nist('Gauss2')


def test_DanWood():  # noqa
    assert_nist('DanWood')


def test_Misra1b():  # noqa
    assert_nist('Misra1b')


def test_Kirby2():  # noqa
    assert_nist('Kirby2')


def test_Hahn1():  # noqa
    assert_nist('Hahn1')


def test_Nelson():  # noqa
    assert_nist('Nelson')


def test_MGH17():  # noqa
    assert_nist('MGH17',
                fit_method=Model.FitMethod.POWELL_LS,
                max_iterations=2200,
                relative=2e-4)


def test_Lanczos1():  # noqa
    assert_nist('Lanczos1',
                max_iterations=1500)


def test_Lanczos2():  # noqa
    assert_nist('Lanczos2',
                max_iterations=1500)


def test_Gauss3():  # noqa
    assert_nist('Gauss3')


def test_Misra1c():  # noqa
    assert_nist('Misra1c')


def test_Misra1d():  # noqa
    assert_nist('Misra1d')


def test_Roszman1():  # noqa
    assert_nist('Roszman1',
                equation='f1',
                relative=4e-6,
                user_dir='./tests/user_defined')


def test_ENSO():  # noqa
    assert_nist('ENSO')


def test_MGH09():  # noqa
    assert_nist('MGH09',
                max_iterations=1900)


def test_Thurber():  # noqa
    assert_nist('Thurber')


def test_BoxBOD():  # noqa
    assert_nist('BoxBOD',
                fit_method=Model.FitMethod.POWELL_LS)


def test_Rat42():  # noqa
    assert_nist('Rat42')


def test_MGH10():  # noqa
    assert_nist('MGH10',
                fit_method=Model.FitMethod.AMOEBA_LS,
                max_iterations=2600,
                show_warnings=False)


def test_Eckerle4():  # noqa
    assert_nist('Eckerle4')


def test_Rat43():  # noqa
    assert_nist('Rat43')


def test_Bennett5():  # noqa
    assert_nist('Bennett5',
                fit_method=Model.FitMethod.AMOEBA_LS,
                max_iterations=6000,
                relative=2e-6,
                show_warnings=False)
