import re

import numpy as np

from msl.nlf import LinearModel, Model


def test_input_uncorrelated() -> None:
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [15.6, 17.5, 36.6, 43.8, 58.2, 61.6, 64.2, 70.4, 98.8]
    with LinearModel() as model:
        params = model.create_parameters([("a1", 5, False, "intercept"), ("a2", 10, False, "slope")])
        got = model.fit(x=x, y=y, params=params, debug=True)

    expected = """Input(
  absolute_residuals=True
  correlated=False
  correlations=
    Correlations(
      data=[]
      is_correlated=[[False False]
                     [False False]]
    )
  delta=0.1
  equation='a1+a2*x'
  fit_method=<FitMethod.LM: 'Levenberg-Marquardt'>
  max_iterations=999
  params=
    InputParameters(
      InputParameter(name='a1', value=5.0, constant=False, label='intercept'),
      InputParameter(name='a2', value=10.0, constant=False, label='slope')
    )
  residual_type=<ResidualType.DY_X: 'dy v x'>
  second_derivs_B=True
  second_derivs_H=True
  tolerance=1e-20
  ux=[[0. 0. 0. 0. 0. 0. 0. 0. 0.]]
  uy=[0. 0. 0. 0. 0. 0. 0. 0. 0.]
  uy_weights_only=False
  weighted=False
  x=[[1. 2. 3. 4. 5. 6. 7. 8. 9.]]
  y=[15.6 17.5 36.6 43.8 58.2 61.6 64.2 70.4 98.8]
)"""
    for g, e in zip(str(got).splitlines(), expected.splitlines()):
        # In case the IDE removes trailing whitespace
        if e == "  correlations=":
            assert g == "  correlations=    "
        elif e == "  params=":
            assert g == "  params=    "
        else:
            assert g == e


def test_input_correlated() -> None:
    x = np.array([[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]])
    y = [1.1, 1.9, 3.2, 3.7]
    a = [0, 0.9, 0]
    uy = [0.5, 0.5, 0.5, 0.5]
    ux = [[0.01, 0.02, 0.03, 0.04], [0.002, 0.004, 0.006, 0.008]]
    with Model("a1+a2*(x+exp(a3*x))+x2", weighted=True, correlated=True) as model:
        model.set_correlation("y", "y", value=0.5)
        model.set_correlation("x", "x", value=0.8)
        got = model.fit(x=x, y=y, params=a, uy=uy, ux=ux, debug=True)

    expected = """Input(
  absolute_residuals=True
  correlated=True
  correlations=
    Correlations(
      data=[
        Correlation(
          coefficients=[[1.  0.8 0.8 0.8]
                        [0.8 1.  0.8 0.8]
                        [0.8 0.8 1.  0.8]
                        [0.8 0.8 0.8 1. ]]
          path=SKIP
        ),
        Correlation(
          coefficients=[[1.  0.5 0.5 0.5]
                        [0.5 1.  0.5 0.5]
                        [0.5 0.5 1.  0.5]
                        [0.5 0.5 0.5 1. ]]
          path=SKIP
        )]
      is_correlated=[[ True False False]
                     [False  True False]
                     [False False False]]
    )
  delta=0.1
  equation='a1+a2*(x+exp(a3*x))+x2'
  fit_method=<FitMethod.LM: 'Levenberg-Marquardt'>
  max_iterations=999
  params=
    InputParameters(
      InputParameter(name='a1', value=0.0, constant=False, label=None),
      InputParameter(name='a2', value=0.9, constant=False, label=None),
      InputParameter(name='a3', value=0.0, constant=False, label=None)
    )
  residual_type=<ResidualType.DY_X: 'dy v x'>
  second_derivs_B=True
  second_derivs_H=True
  tolerance=1e-20
  ux=[[0.01  0.02  0.03  0.04 ]
      [0.002 0.004 0.006 0.008]]
  uy=[0.5 0.5 0.5 0.5]
  uy_weights_only=False
  weighted=True
  x=[[1.  2.  3.  4. ]
     [0.1 0.2 0.3 0.4]]
  y=[1.1 1.9 3.2 3.7]
)"""
    for g, e in zip(str(got).splitlines(), expected.splitlines()):
        if "SKIP" in e:
            continue

        # In case the IDE removes trailing whitespace
        if e == "  correlations=":
            assert g == "  correlations=    "
        elif e == "  params=":
            assert g == "  params=    "
        elif e == "      data=[":
            assert g == "      data=[    "
        else:
            assert g == e


def test_result_1() -> None:
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [15.6, 17.5, 36.6, 43.8, 58.2, 61.6, 64.2, 70.4, 98.8]
    with LinearModel() as model:
        params = model.create_parameters([("a1", 5, False, "intercept"), ("a2", 10, False, "slope")])
        got = model.fit(x=x, y=y, params=params)

    expected = """Result(
  chisq=316.6580555555554
  correlation=[[ 1.         -0.88852332]
               [-0.88852332  1.        ]]
  covariance=[[ 0.52777778 -0.08333333]
              [-0.08333333  0.01666667]]
  dof=7.0
  eof=6.725835641715091
  iterations=20
  num_calls=2
  params=
    ResultParameters(
      ResultParameter(name='a1', value=4.813888860975824, uncert=4.8862063121833526, label='intercept'),
      ResultParameter(name='a2', value=9.408333338293616, uncert=0.8683016476563606, label='slope')
    )
)"""
    for g, e in zip(str(got).splitlines(), expected.splitlines()):
        # In case the IDE removes trailing whitespace
        if e == "  params=":
            assert g == "  params=    "
        else:
            assert g == e


def test_result_2() -> None:
    x = np.array([[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]])
    y = np.array([1.1, 1.9, 3.2, 3.7])
    a = np.array([0, 0.9, 0])
    uy = np.array([0.5, 0.5, 0.5, 0.5])
    ux = np.array([[0.01, 0.02, 0.03, 0.04], [0.002, 0.004, 0.006, 0.008]])
    with Model("a1+a2*(x+exp(a3*x))+x2", weighted=True, correlated=True) as model:
        model.set_correlation("y", "y", value=0.5)
        model.set_correlation("x", "x", value=0.8)
        got = model.fit(x=x, y=y, params=a, uy=uy, ux=ux)

    expected = r"""Result\(
  chisq=0\.85487560020565\d{2}
  correlation=\[\[ 1\.         -0\.81341696  0\.33998683\]
               \[-0\.81341696  1\.         -0\.41807236\]
               \[ 0\.33998683 -0\.41807236  1\.        \]\]
  covariance=\[\[ 4\.62857806e-01 -8\.76652826e-02  2\.75368388e-05\]
              \[-8\.76652826e-02  2\.50946556e-02 -7\.88442937e-06\]
              \[ 2\.75368388e-05 -7\.88442937e-06  1\.41728236e-08\]\]
  dof=inf
  eof=0\.32710857899179\d{2}
  iterations=33
  num_calls=3
  params=\s{4}
    ResultParameters\(
      ResultParameter\(name='a1', value=-0\.61018807476402\d{2}, uncert=0\.68033653854569\d{2}, label=None\),
      ResultParameter\(name='a2', value=0\.81002888697772\d{2}, uncert=0\.158412927425664\d{1,2}, label=None\),
      ResultParameter\(name='a3', value=4\.585005881\d{6}e-05, uncert=0\.00011904966869\d{6}, label=None\)
    \)
\)"""
    for g, e in zip(str(got).splitlines(), expected.splitlines()):
        assert re.match(e, g) is not None
