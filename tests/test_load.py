import os

import numpy as np

from msl.nlf import load
from msl.nlf import version_info


def get_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), 'nlf', f'{filename}.nlf')


def test_load_3_0():
    loaded = load(get_path('3_0'))
    assert loaded.version() == f'{version_info.major}.{version_info.minor}'
    assert loaded.nlf_version == '3.0'
    assert loaded.nlf_path.endswith('3_0.nlf')
    assert loaded.equation == 'a3/(exp(a5/(a1*(1-6*a2^2/a1^2)*(x+273.15)+a5/2*a2^2/a1^2))-a4)'
    assert loaded.num_variables == 1
    assert loaded.num_parameters == 5
    assert loaded.comments == ''

    inputs = loaded.fit(loaded.x, loaded.y, params=loaded.params,
                        ux=loaded.ux, uy=loaded.uy, debug=True)
    assert inputs.correlated is False
    assert inputs.weighted is True
    assert inputs.max_iterations == 999
    assert inputs.tolerance == 1E-20
    assert inputs.delta == 0.1
    assert inputs.fitting_method == 'Levenberg-Marquardt'
    assert inputs.second_derivs_H is True
    assert inputs.second_derivs_B is True
    assert inputs.uy_weights_only is False

    params = loaded.params
    assert len(params) == 5
    assert params['a1'].value == 9.6715769163976862e-7
    assert params['a1'].constant is False
    assert params['a1'].label is None
    assert params['a2'].value == 5.204137289790214e-8
    assert params['a2'].constant is False
    assert params['a2'].label is None
    assert params['a3'].value == 2.0347598692887630e-2
    assert params['a3'].constant is False
    assert params['a3'].label is None
    assert params['a4'].value == 1.0000000000000000e0
    assert params['a4'].constant is True
    assert params['a4'].label is None
    assert params['a5'].value == 1.4388000000000000e-2
    assert params['a5'].constant is True
    assert params['a5'].label is None

    x = np.array([1000, 1020, 1040, 1060, 1080, 1100, 1120, 1140, 1160, 1180,
                  1200, 1220, 1240, 1260, 1280, 1300, 1320, 1340, 1360, 1380,
                  1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560, 1580,
                  1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780,
                  1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980,
                  2000], dtype=float)

    y = np.array(
        [1.7040959993392E-07, 2.0357312143118E-07, 2.4189735781301E-07, 2.8597553819474E-07,
         3.3644221784967E-07, 3.9397361034069E-07, 4.5928774393274E-07, 5.3314444375737E-07,
         6.1634514266271E-07, 7.0973252500909E-07, 8.1419000885568E-07, 9.3064107302532E-07,
         1.0600484364332E-06, 1.2034130978221E-06, 1.3617732446679E-06, 1.5362030405084E-06,
         1.7278113003169E-06, 1.9377400637897E-06, 2.1671630765644E-06, 2.4172841894298E-06,
         2.6893356855482E-06, 2.9845765455875E-06, 3.3042906604721E-06, 3.6497850012071E-06,
         4.0223877549288E-06, 4.4234464359823E-06, 4.8543259804447E-06, 5.3164068320968E-06,
         5.8110830274089E-06, 6.3397602866522E-06, 6.9038541177828E-06, 7.5047879392748E-06,
         8.1439912276048E-06, 8.8228976946216E-06, 9.5429434995687E-06, 1.0305565500074E-05,
         1.1112199545971E-05, 1.1964278819401E-05, 1.2863232224196E-05, 1.3810482827192E-05,
         1.4807446353699E-05, 1.5855529739011E-05, 1.6956129737494E-05, 1.8110631590474E-05,
         1.9320407753823E-05, 2.0586816685884E-05, 2.1911201696092E-05, 2.3294889854414E-05,
         2.4739190961502E-05, 2.6245396579253E-05, 2.7814779121266E-05])

    uy = np.array(
        [1.7040959993392E-07, 2.0357312143118E-07, 2.4189735781301E-07, 2.8597553819474E-07,
         3.3644221784967E-07, 3.9397361034069E-07, 4.5928774393274E-07, 5.3314444375737E-07,
         6.1634514266271E-07, 7.0973252500909E-07, 8.1419000885568E-07, 9.3064107302532E-07,
         1.0600484364332E-06, 1.2034130978221E-06, 1.3617732446679E-06, 1.5362030405084E-06,
         1.7278113003169E-06, 1.9377400637897E-06, 2.1671630765644E-06, 2.4172841894298E-06,
         2.6893356855482E-06, 2.9845765455875E-06, 3.3042906604721E-06, 3.6497850012071E-06,
         4.0223877549288E-06, 4.4234464359823E-06, 4.8543259804447E-06, 5.3164068320968E-06,
         5.8110830274089E-06, 6.3397602866522E-06, 6.9038541177828E-06, 7.5047879392748E-06,
         8.1439912276048E-06, 8.8228976946216E-06, 9.5429434995687E-06, 1.0305565500074E-05,
         1.1112199545971E-05, 1.1964278819401E-05, 1.2863232224196E-05, 1.3810482827192E-05,
         1.4807446353699E-05, 1.5855529739011E-05, 1.6956129737494E-05, 1.8110631590474E-05,
         1.9320407753823E-05, 2.0586816685884E-05, 2.1911201696092E-05, 2.3294889854414E-05,
         2.4739190961502E-05, 2.6245396579253E-05, 2.7814779121266E-05])

    assert np.array_equal(loaded.x, x)
    assert loaded.x.shape == loaded.ux.shape
    assert np.sum(loaded.ux) == 0.0
    assert np.allclose(loaded.y, y, rtol=1e-30, atol=1.e-20)
    assert np.allclose(loaded.uy, uy, rtol=1e-30, atol=1.e-20)


def test_5_41():
    loaded = load(get_path('5_41'))
    assert loaded.version() == f'{version_info.major}.{version_info.minor}'
    assert loaded.nlf_version == '5.41'
    assert loaded.nlf_path.endswith('5_41.nlf')
    assert loaded.equation == 'a1+a2*(x+exp(a3*x))+x2'
    assert loaded.num_variables == 2
    assert loaded.num_parameters == 3
    assert loaded.comments == 'Correlated and \r\nweighted example'

    inputs = loaded.fit(loaded.x, loaded.y, params=loaded.params,
                        ux=loaded.ux, uy=loaded.uy, debug=True)
    assert inputs.correlated is True
    assert inputs.weighted is True
    assert inputs.max_iterations == 987
    assert inputs.tolerance == 1.4e-19
    assert inputs.delta == 0.12
    assert inputs.fitting_method == 'Levenberg-Marquardt'
    assert inputs.second_derivs_H is True
    assert inputs.second_derivs_B is False
    assert inputs.uy_weights_only is True

    y_y_corr = np.array([[1.0, 0.5, 0.5, 0.5],
                         [0.5, 1.0, 0.5, 0.5],
                         [0.5, 0.5, 1.0, 0.5],
                         [0.5, 0.5, 0.5, 1.0]])

    y_x1_corr = np.array([[1.0, 0.1, 0.1, 0.1],
                          [0.1, 1.0, 0.1, 0.1],
                          [0.1, 0.1, 1.0, 0.1],
                          [0.1, 0.1, 0.1, 1.0]])

    x1_x1_corr = np.array([[1.0, 0.8, 0.8, 0.8],
                           [0.8, 1.0, 0.8, 0.8],
                           [0.8, 0.8, 1.0, 0.8],
                           [0.8, 0.8, 0.8, 1.0]])

    is_correlated = inputs.correlations.is_correlated
    assert np.array_equal(is_correlated, [[True, True, False],
                                          [True, True, False],
                                          [False, False, False]])

    for data in inputs.correlations.data:
        if data.path.endswith('Y-Y.txt'):
            assert np.array_equal(y_y_corr, data.coefficients)
        elif data.path.endswith('Y-X1.txt') or data.path.endswith('X1-Y.txt'):
            assert np.array_equal(y_x1_corr, data.coefficients)
        elif data.path.endswith('X1-X1.txt'):
            assert np.array_equal(x1_x1_corr, data.coefficients)
        else:
            raise ValueError(f'Could not find correction identifier in {data.path!r}')

    params = loaded.params
    assert len(params) == 3
    assert params['a1'].value == -6.1000001267144628e-1
    assert params['a1'].constant is False
    assert params['a1'].label is None
    assert params['a2'].value == 8.1000001365243869e-1
    assert params['a2'].constant is True
    assert params['a2'].label is None
    assert params['a3'].value == -1.7332607114236678e-8
    assert params['a3'].constant is False
    assert params['a3'].label is None

    x = np.array([[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]], dtype=float)
    ux = np.array([[0.01, 0.02, 0.03, 0.04], [0.002, 0.004, 0.006, 0.008]])
    y = np.array([1.1, 1.9, 3.2, 3.7])
    uy = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.array_equal(loaded.x, x)
    assert np.array_equal(loaded.ux, ux)
    assert np.array_equal(loaded.y, y)
    assert np.array_equal(loaded.uy, uy)
