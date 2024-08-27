import pytest

from msl.nlf import Model


@pytest.mark.parametrize("xs", ["x", "x1"])
def test_log_bracket_x_issue(xs: str) -> None:
    # Keith reported a "RuntimeError: Error calculating derivatives for parameter covariances"
    # for the following model. Ensure that the fit now succeeds.
    with Model(f"a1+a2*{xs}+a3*log({xs})", weighted=True) as model:
        params = [-1.2643837268033873, -0.0005665840259614689, 0.34589838624460595]
        ux=[0., 0., 0., 0., 0., 0., 0.]
        uy=[0.0584211, 0.02414976, 0.02222908, 0.01654807, 0.01195032, 0.01123268, 0.01284842]
        x=[5., 10., 20., 40., 60., 100., 120.]
        y=[-1.05256713, -0.88937887, -0.80973676, -0.74631119, -0.70289318, -0.63192922, -0.60112008]

        results = model.fit(x, y, ux=ux, uy=uy, params=params)
        assert pytest.approx(results.params["a1"].value, abs=2e-8) == -1.14171093389872
        assert pytest.approx(results.params["a1"].uncert, abs=1e-9) == 0.0639731504340679
        assert pytest.approx(results.params["a2"].value, abs=1e-11) == 0.000350958449375458
        assert pytest.approx(results.params["a2"].uncert, abs=1e-10) == 0.000461983374717262
        assert pytest.approx(results.params["a3"].value, abs=1e-8) == 0.238002933414618
        assert pytest.approx(results.params["a3"].uncert, abs=1e-9) == 0.0530346733839314
        assert pytest.approx(results.chisq, abs=5e-7) == 2.80294335418112
        assert pytest.approx(results.eof, abs=1e-9) == 0.0407179312369875
