from msl.nlf import LinearModel
from msl.nlf import Model


def test_input_uncorrelated():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [15.6, 17.5, 36.6, 43.8, 58.2, 61.6, 64.2, 70.4, 98.8]
    with LinearModel() as model:
        params = model.create_parameters(
            [('a1', 5, False, 'intercept'),
             ('a2', 10, False, 'slope')])
        got = model.fit(x=x, y=y, params=params, debug=True)

    expected = """Input(
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
        assert g == e


def test_input_correlated():
    x = [[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]]
    y = [1.1, 1.9, 3.2, 3.7]
    a = [0, 0.9, 0]
    sigy = [0.5, 0.5, 0.5, 0.5]
    sigx = [[0.01, 0.02, 0.03, 0.04], [0.002, 0.004, 0.006, 0.008]]
    with Model('a1+a2*(x+exp(a3*x))+x2', weighted=True, correlated=True) as model:
        model.set_correlation('y', 'y', value=0.5)
        model.set_correlation('x', 'x', value=0.8)
        got = model.fit(x=x, y=y, params=a, uy=sigy, ux=sigx, debug=True)

    expected = """Input(
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
        if 'SKIP' in e:
            continue
        assert g == e


def test_result_1():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [15.6, 17.5, 36.6, 43.8, 58.2, 61.6, 64.2, 70.4, 98.8]
    with LinearModel() as model:
        params = model.create_parameters([('a1', 5, False, 'intercept'),
                                          ('a2', 10, False, 'slope')])
        got = model.fit(x=x, y=y, params=params)

    expected = """Result(
  calls=2
  chisq=316.6580555555554
  correlation=[[ 1.         -0.88852332]
               [-0.88852332  1.        ]]
  covariance=[[ 0.52777778 -0.08333333]
              [-0.08333333  0.01666667]]
  dof=7
  eof=6.725835641715091
  iterations=20
  params=    
    ResultParameters(
      ResultParameter(name='a1', value=4.813888860975824, uncert=4.8862063121833526, label='intercept'),
      ResultParameter(name='a2', value=9.408333338293616, uncert=0.8683016476563606, label='slope')
    )
)"""
    for g, e in zip(str(got).splitlines(), expected.splitlines()):
        assert g == e


def test_result_2():
    x = [[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]]
    y = [1.1, 1.9, 3.2, 3.7]
    a = [0, 0.9, 0]
    sigy = [0.5, 0.5, 0.5, 0.5]
    sigx = [[0.01, 0.02, 0.03, 0.04], [0.002, 0.004, 0.006, 0.008]]
    with Model('a1+a2*(x+exp(a3*x))+x2', weighted=True, correlated=True) as model:
        model.set_correlation('y', 'y', value=0.5)
        model.set_correlation('x', 'x', value=0.8)
        got = model.fit(x=x, y=y, params=a, uy=sigy, ux=sigx)

    expected = """Result(
  calls=3
  chisq=0.854875600205648
  correlation=[[ 1.         -0.81341696  0.33998683]
               [-0.81341696  1.         -0.41807236]
               [ 0.33998683 -0.41807236  1.        ]]
  covariance=[[ 4.62857806e-01 -8.76652826e-02  2.75368388e-05]
              [-8.76652826e-02  2.50946556e-02 -7.88442937e-06]
              [ 2.75368388e-05 -7.88442937e-06  1.41728236e-08]]
  dof=1
  eof=0.32710857899179385
  iterations=33
  params=    
    ResultParameters(
      ResultParameter(name='a1', value=-0.6101880747640294, uncert=0.6803365385456976, label=None),
      ResultParameter(name='a2', value=0.8100288869777268, uncert=0.1584129274256673, label=None),
      ResultParameter(name='a3', value=4.585005881907852e-05, uncert=0.00011904966869376515, label=None)
    )
)"""
    for g, e in zip(str(got).splitlines(), expected.splitlines()):
        assert g == e
