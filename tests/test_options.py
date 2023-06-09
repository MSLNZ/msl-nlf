import pytest

from msl.nlf import LinearModel


def test_invalid_kwarg():
    with pytest.raises(TypeError, match=r'unexpected keyword argument'):
        LinearModel(invalid=True)

    with pytest.raises(TypeError, match=r'unexpected keyword argument'):
        LinearModel().options(invalid=True)

    with pytest.raises(TypeError, match=r'1 positional argument'):
        LinearModel().options(True)


@pytest.mark.parametrize(
    'method',
    [LinearModel.FitMethod.LM,
     'LM',
     'Levenberg-Marquardt',
     LinearModel.FitMethod.AMOEBA_LS,
     'AMOEBA_LS',
     'Amoeba least squares',
     LinearModel.FitMethod.AMOEBA_MD,
     'AMOEBA_MD',
     'Amoeba minimum distance',
     LinearModel.FitMethod.AMOEBA_MM,
     'AMOEBA_MM',
     'Amoeba minimax',
     LinearModel.FitMethod.POWELL_LS,
     'POWELL_LS',
     'Powell least squares',
     LinearModel.FitMethod.POWELL_MD,
     'POWELL_MD',
     'Powell minimum distance',
     LinearModel.FitMethod.POWELL_MM,
     'POWELL_MM',
     'Powell minimax',
     ])
def test_method_valid(method):
    # all these data types and values for the 'method' kwarg are valid
    with LinearModel() as model:
        model.options(method=method)


@pytest.mark.parametrize(
    'method',
    ['',
     'LevMar',
     'lev-mar',
     'Powell',
     'Amoeba mini',
     'powell least squares',
     2,
     None])
def test_method_invalid(method):
    # all these data types and values for the 'method' kwarg are invalid
    with LinearModel() as model:
        with pytest.raises(ValueError, match=r'enum member name or value'):
            model.options(method=method)
