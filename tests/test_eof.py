from __future__ import annotations

import numpy as np
import pytest

from msl.nlf import FitMethod, LinearModel, ResidualType


@pytest.mark.parametrize(
    ("abs_res", "res_typ", "method", "expected"),
    [
        (True, ResidualType.DY_X, FitMethod.LM, 1.20048099143212),
        (False, ResidualType.DY_X, FitMethod.LM, 0.0623630090362921),
        (True, ResidualType.DX_X, FitMethod.LM, 11.387292701655),
        (False, ResidualType.DX_X, FitMethod.LM, 0.075933882253817),
        (True, ResidualType.DY_Y, FitMethod.LM, 1.20048099143212),
        (False, ResidualType.DY_Y, FitMethod.LM, 0.0623630090362921),
        (True, ResidualType.DX_Y, FitMethod.LM, 11.387292701655),
        (False, ResidualType.DX_Y, FitMethod.LM, 0.075933882253817),
        (True, ResidualType.DY_X, FitMethod.AMOEBA_LS, 1.20048099143212),
        (True, ResidualType.DY_X, FitMethod.AMOEBA_MD, 1.2004811773973),
        (True, ResidualType.DY_X, FitMethod.AMOEBA_MM, None),
        (True, ResidualType.DY_X, FitMethod.POWELL_LS, 1.20048099143212),
        (True, ResidualType.DY_X, FitMethod.POWELL_MD, 1.2004811773973),
        (True, ResidualType.DY_X, FitMethod.POWELL_MM, None),
        (False, ResidualType.DY_X, FitMethod.AMOEBA_LS, 0.0623630090881956),
        (False, ResidualType.DY_X, FitMethod.AMOEBA_MD, 0.0623347441743656),
        (False, ResidualType.DY_X, FitMethod.AMOEBA_MM, None),
        (False, ResidualType.DY_X, FitMethod.POWELL_LS, 0.0623630090462169),
        (False, ResidualType.DY_X, FitMethod.POWELL_MD, 0.062334744153301),
        (False, ResidualType.DY_X, FitMethod.POWELL_MM, None),
        (True, ResidualType.DX_X, FitMethod.AMOEBA_LS, 11.3872927020648),
        (True, ResidualType.DX_X, FitMethod.AMOEBA_MD, 11.3869735190884),
        (True, ResidualType.DX_X, FitMethod.POWELL_LS, 11.38729270172),
        (True, ResidualType.DX_X, FitMethod.POWELL_MD, 11.3869735190934),
        (False, ResidualType.DX_X, FitMethod.AMOEBA_LS, 0.075933882321313),
        (False, ResidualType.DX_X, FitMethod.AMOEBA_MD, 0.0758965670178389),
        (False, ResidualType.DX_X, FitMethod.POWELL_LS, 0.0759338822666471),
        (False, ResidualType.DX_X, FitMethod.POWELL_MD, 0.075896566991427),
    ],
)
def test_eof(abs_res: bool, res_typ: ResidualType, method: FitMethod, expected: float | None) -> None:  # noqa: FBT001
    # The expected value comes from the Delphi GUI
    #
    # For MiniMax fitting, the GUI returns the maximum residual
    # whereas the NLF library returns the error of fit (therefore we calculate the expected value below)

    x = np.array([77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0])
    y = np.array([10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.4, 75.47, 81.78])
    params = np.array([1.0, 0.1])

    with LinearModel() as model:
        model.show_warnings = False
        model.options(absolute_residuals=abs_res, residual_type=res_typ, fit_method=method)
        result = model.fit(x, y, params=params)
        if expected is None:
            residuals = y - model.evaluate(x, result)
            if not abs_res:
                residuals /= y
            chisq = np.sum(np.square(residuals))
            eof = np.sqrt(chisq / (len(y) - len(params)))
            assert pytest.approx(eof, rel=1e-10) == result.eof
        else:
            assert pytest.approx(expected, rel=1e-7) == result.eof
