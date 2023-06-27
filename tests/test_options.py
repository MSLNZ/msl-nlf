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
def test_fit_method_valid(method):
    # all these data types and values for the 'fit_method' kwarg are valid
    with LinearModel() as model:
        model.options(fit_method=method)


@pytest.mark.parametrize(
    'residual_type',
    [LinearModel.ResidualType.DX_X,
     'DX_X',
     'dx v x',
     LinearModel.ResidualType.DX_Y,
     'DX_Y',
     'dx v y',
     LinearModel.ResidualType.DY_X,
     'DY_X',
     'dy v x',
     LinearModel.ResidualType.DY_Y,
     'DY_Y',
     'dy v y'
     ])
def test_residual_type_valid(residual_type):
    # all these data types and values for the 'residual_type' kwarg are valid
    with LinearModel() as model:
        model.options(residual_type=residual_type)


@pytest.mark.parametrize(
    'method',
    ['',
     'LevMar',
     'lev-mar',
     'Powell',
     'Amoeba mini',
     'powell least squares',
     2])
def test_fit_method_invalid(method):
    # all these data types and values for the 'fit_method' kwarg are invalid
    with LinearModel() as model:
        with pytest.raises(ValueError, match=r'FitMethod enum member name or value'):
            model.options(fit_method=method)


@pytest.mark.parametrize(
    'residual_type',
    ['',
     'uy_y',
     'Levenberg-Marquardt',
     'DY Y',
     'something',
     2])
def test_residual_type_invalid(residual_type):
    # all these data types and values for the 'residual_type' kwarg are invalid
    with LinearModel() as model:
        with pytest.raises(ValueError, match=r'ResidualType enum member name or value'):
            model.options(residual_type=residual_type)


def test_default():
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    params = [0, 1]
    with LinearModel() as model:
        inputs = model.fit(x, y, params=params, debug=True)
        assert inputs.absolute_residuals is True
        assert inputs.correlated is False
        assert inputs.delta == 0.1
        assert inputs.fit_method == model.FitMethod.LM
        assert inputs.max_iterations == 999
        assert inputs.residual_type == model.ResidualType.DY_X
        assert inputs.second_derivs_B is True
        assert inputs.second_derivs_H is True
        assert inputs.tolerance == 1e-20
        assert inputs.uy_weights_only is False
        assert inputs.weighted is False

        model.options(second_derivs_B=False, max_iterations=123)
        inputs = model.fit(x, y, params=params, debug=True)
        assert inputs.absolute_residuals is True
        assert inputs.correlated is False
        assert inputs.delta == 0.1
        assert inputs.fit_method == model.FitMethod.LM
        assert inputs.max_iterations == 123
        assert inputs.residual_type == model.ResidualType.DY_X
        assert inputs.second_derivs_B is False
        assert inputs.second_derivs_H is True
        assert inputs.tolerance == 1e-20
        assert inputs.uy_weights_only is False
        assert inputs.weighted is False

        model.options(weighted=True)
        inputs = model.fit(x, y, params=params, debug=True)
        assert inputs.absolute_residuals is True
        assert inputs.correlated is False
        assert inputs.delta == 0.1
        assert inputs.fit_method == model.FitMethod.LM
        assert inputs.max_iterations == 123
        assert inputs.residual_type == model.ResidualType.DY_X
        assert inputs.second_derivs_B is False
        assert inputs.second_derivs_H is True
        assert inputs.tolerance == 1e-20
        assert inputs.uy_weights_only is False
        assert inputs.weighted is True

        model.options(correlated=True, fit_method=model.FitMethod.POWELL_MD, delta=0)
        inputs = model.fit(x, y, params=params, debug=True)
        assert inputs.absolute_residuals is True
        assert inputs.correlated is True
        assert inputs.delta == 0.0
        assert inputs.fit_method == model.FitMethod.POWELL_MD
        assert inputs.max_iterations == 123
        assert inputs.residual_type == model.ResidualType.DY_X
        assert inputs.second_derivs_B is False
        assert inputs.second_derivs_H is True
        assert inputs.tolerance == 1e-20
        assert inputs.uy_weights_only is False
        assert inputs.weighted is True

        model.options(fit_method='AMOEBA_LS', residual_type='dy v y')
        inputs = model.fit(x, y, params=params, debug=True)
        assert inputs.absolute_residuals is True
        assert inputs.correlated is True
        assert inputs.delta == 0.0
        assert inputs.fit_method == model.FitMethod.AMOEBA_LS
        assert inputs.max_iterations == 123
        assert inputs.residual_type == model.ResidualType.DY_Y
        assert inputs.second_derivs_B is False
        assert inputs.second_derivs_H is True
        assert inputs.tolerance == 1e-20
        assert inputs.uy_weights_only is False
        assert inputs.weighted is True

        model.options(fit_method='Amoeba minimax', absolute_residuals=False)
        inputs = model.fit(x, y, params=params, debug=True)
        assert inputs.absolute_residuals is False
        assert inputs.correlated is True
        assert inputs.delta == 0.0
        assert inputs.fit_method == model.FitMethod.AMOEBA_MM
        assert inputs.max_iterations == 123
        assert inputs.residual_type == model.ResidualType.DY_Y
        assert inputs.second_derivs_B is False
        assert inputs.second_derivs_H is True
        assert inputs.tolerance == 1e-20
        assert inputs.uy_weights_only is False
        assert inputs.weighted is True

        model.options(tolerance=1, second_derivs_H=False, uy_weights_only=True)
        inputs = model.fit(x, y, params=params, debug=True)
        assert inputs.absolute_residuals is False
        assert inputs.correlated is True
        assert inputs.delta == 0.0
        assert inputs.fit_method == model.FitMethod.AMOEBA_MM
        assert inputs.max_iterations == 123
        assert inputs.residual_type == model.ResidualType.DY_Y
        assert inputs.second_derivs_B is False
        assert inputs.second_derivs_H is False
        assert inputs.tolerance == 1.0
        assert inputs.uy_weights_only is True
        assert inputs.weighted is True
