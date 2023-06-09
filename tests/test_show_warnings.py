import warnings

import pytest

from msl.nlf import LinearModel


def test_fit_weighted():
    model = LinearModel()

    assert model.show_warnings is True

    # check with ux
    with pytest.warns(UserWarning, match='uncertainties are specified'):
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1],
                  debug=True, weighted=False, ux=[0.1, 0.2, 0.3])

    # UserWarning is suppressed
    model.show_warnings = False
    assert model.show_warnings is False
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1],
                  debug=True, weighted=False, ux=[0.1, 0.2, 0.3])

    # check with uy
    model.show_warnings = True
    assert model.show_warnings is True
    with pytest.warns(UserWarning, match='uncertainties are specified'):
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1],
                  debug=True, weighted=False, uy=[0.1, 0.2, 0.3])
