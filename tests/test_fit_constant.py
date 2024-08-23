import sys

import numpy as np
import pytest

from msl.nlf import LinearModel, PolynomialModel

win32s = [False, True] if sys.platform == "win32" else [False]

rtol = 1e-11
atol = 1e-15


@pytest.mark.parametrize("win32", win32s)
def test_ua_zero(win32: bool) -> None:  # noqa: FBT001
    # In the Delphi code, the software doesn't attempt to calculate a value
    # for ua[i] and it puts a blank cell into the Results spreadsheet. But
    # if a fit had previously been carried out with a[i] varying, then ua[i]
    # will have the previous value because that part of the covariance matrix
    # doesn't get overwritten.

    x = [1.6, 3.2, 5.5, 7.8, 9.4]
    y = [7.8, 19.1, 17.6, 33.9, 45.4]

    with LinearModel(win32=win32) as model:
        params = model.create_parameters()
        params["a1"] = 0
        params["a2"] = 1
        result = model.fit(x, y, params=params)
        assert pytest.approx(0.522439025564517) == result.params["a1"].value
        assert pytest.approx(5.13241814994003) == result.params["a1"].uncert
        assert pytest.approx(4.40682926810331) == result.params["a2"].value
        assert pytest.approx(0.82770172450896) == result.params["a2"].uncert
        assert pytest.approx(84.266087804878) == result.chisq
        assert pytest.approx(5.29987697356829) == result.eof
        assert np.allclose(
            [[1.0, -0.886981409504304], [-0.886981409504304, 1.0]], result.correlation, rtol=rtol, atol=atol
        )

        params["a1"].constant = True
        result = model.fit(x, y, params=params)
        assert result.params["a1"].value == 0.0
        assert result.params["a1"].uncert == 0.0
        assert pytest.approx(4.48156046796625) == result.params["a2"].value
        assert pytest.approx(0.331598037682679) == result.params["a2"].uncert
        assert pytest.approx(84.5571318595579) == result.chisq
        assert pytest.approx(4.59774759690976) == result.eof
        assert np.allclose([[0.0, 0.0], [0.0, 1.0]], result.correlation, rtol=rtol, atol=atol)

        params["a1"].value = 0.1
        params["a1"].constant = True
        params["a2"].value = 5.0
        params["a2"].constant = True
        result = model.fit(x, y, params=params)
        assert result.params["a1"].value == 0.1
        assert result.params["a1"].uncert == 0.0
        assert result.params["a2"].value == 5.0
        assert result.params["a2"].uncert == 0.0
        assert pytest.approx(139.02) == result.chisq
        assert pytest.approx(5.27294983856285) == result.eof
        assert np.allclose([[0.0, 0.0], [0.0, 0.0]], result.correlation, rtol=rtol, atol=atol)

        params["a1"].constant = False
        params["a2"].constant = False
        result = model.fit(x, y, params=params)
        assert pytest.approx(0.522439025564517) == result.params["a1"].value
        assert pytest.approx(5.13241814994003) == result.params["a1"].uncert
        assert pytest.approx(4.40682926810331) == result.params["a2"].value
        assert pytest.approx(0.82770172450896) == result.params["a2"].uncert
        assert pytest.approx(84.266087804878) == result.chisq
        assert pytest.approx(5.29987697356829) == result.eof
        assert np.allclose(
            [[1.0, -0.886981409504304], [-0.886981409504304, 1.0]], result.correlation, rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("win32", win32s)
def test_intermixed(win32: bool) -> None:  # noqa: FBT001, PLR0915
    # intermix constant=True and constant=False
    x = np.linspace(0, 10)
    y = np.zeros(x.size)
    for i, c in enumerate([1.2, 0.2, 0.04, 0.005, 0.00071, 0.0000912]):
        y += c * x**i

    with PolynomialModel(5, win32=win32) as model:
        params = model.create_parameters()
        params["a1"] = 1
        params["a2"] = 0.21, True
        params["a3"] = 1
        params["a4"] = 0.0048, True
        params["a5"] = 1
        params["a6"] = 1
        result = model.fit(x, y, params=params)
        assert pytest.approx(1.19179154233815) == result.params["a1"].value
        assert pytest.approx(0.00080619267473275) == result.params["a1"].uncert
        assert result.params["a2"].value == 0.21
        assert result.params["a2"].uncert == 0.0
        assert pytest.approx(0.0379692475134135) == result.params["a3"].value
        assert pytest.approx(0.000111326886065598) == result.params["a3"].uncert
        assert result.params["a4"].value == 0.0048
        assert result.params["a4"].uncert == 0.0
        assert pytest.approx(0.000784146045962053) == result.params["a5"].value
        assert pytest.approx(4.60180851979069e-6) == result.params["a5"].uncert
        assert pytest.approx(8.68640395813181e-5) == result.params["a6"].value
        assert pytest.approx(3.69045503059371e-7) == result.params["a6"].uncert
        assert pytest.approx(0.000307031412049968) == result.chisq
        assert pytest.approx(0.0025835239333379) == result.eof
        assert np.allclose(
            [
                [1.0, 0.0, -0.73707245815978, 0.0, 0.5905724259644, -0.543436954534029],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.73707245815978, 0.0, 1.0, 0.0, -0.957967695532247, 0.926545146588732],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5905724259644, 0.0, -0.957967695532247, 0.0, 1.0, -0.994977962978252],
                [-0.543436954534029, 0.0, 0.926545146588732, 0.0, -0.994977962978252, 1.0],
            ],
            result.correlation,
            rtol=rtol,
            atol=atol,
        )

        params = model.create_parameters()
        params["a1"] = 1.18, True
        params["a2"] = 1
        params["a3"] = 0.42, True
        params["a4"] = 1
        params["a5"] = 0.00067, True
        params["a6"] = 0.00009, True
        result = model.fit(x, y, params=params)
        assert result.params["a1"].value == 1.18
        assert result.params["a1"].uncert == 0.0
        assert pytest.approx(-1.00304760142689) == result.params["a2"].value
        assert pytest.approx(0.0458161747416433) == result.params["a2"].uncert
        assert result.params["a3"].value == 0.42
        assert result.params["a3"].uncert == 0.0
        assert pytest.approx(-0.0219015200658383) == result.params["a4"].value
        assert pytest.approx(0.000685996551302552) == result.params["a4"].uncert
        assert result.params["a5"].value == 0.00067
        assert result.params["a5"].uncert == 0.0
        assert result.params["a6"].value == 9e-5
        assert result.params["a6"].uncert == 0.0
        assert pytest.approx(27.1235404837317) == result.chisq
        assert pytest.approx(0.7517138817913) == result.eof
        assert np.allclose(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -0.916577493321411, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.916577493321411, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            result.correlation,
            rtol=rtol,
            atol=atol,
        )

        params = model.create_parameters()
        params["a1"] = 1
        params["a2"] = 0.19, True
        params["a3"] = 0.42, True
        params["a4"] = 1
        params["a5"] = 1
        params["a6"] = 1
        result = model.fit(x, y, params=params)
        assert pytest.approx(0.768218603923503) == result.params["a1"].value
        assert pytest.approx(0.0600100402968086) == result.params["a1"].uncert
        assert result.params["a2"].value == 0.19
        assert result.params["a2"].uncert == 0.0
        assert result.params["a3"].value == 0.42
        assert result.params["a3"].uncert == 0.0
        assert pytest.approx(-0.145373403611101) == result.params["a4"].value
        assert pytest.approx(0.0037469943877699) == result.params["a4"].uncert
        assert pytest.approx(0.0206785790339654) == result.params["a5"].value
        assert pytest.approx(0.00087812660627357) == result.params["a5"].uncert
        assert pytest.approx(-0.000780023028161666) == result.params["a6"].value
        assert pytest.approx(5.19805100610938e-5) == result.params["a6"].uncert
        assert pytest.approx(2.13092728973454) == result.chisq
        assert pytest.approx(0.215231285594304) == result.eof
        assert np.allclose(
            [
                [1.0, 0.0, 0.0, -0.654142925415893, 0.592547042434188, -0.545403959138515],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.654142925415893, 0.0, 0.0, 1.0, -0.992125906324946, 0.974932442126446],
                [0.592547042434188, 0.0, 0.0, -0.992125906324946, 1.0, -0.994983217888929],
                [-0.545403959138515, 0.0, 0.0, 0.974932442126446, -0.994983217888929, 1.0],
            ],
            result.correlation,
            rtol=rtol,
            atol=atol,
        )
