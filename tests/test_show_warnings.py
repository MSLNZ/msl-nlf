import os
import shutil
import tempfile
import warnings

import numpy as np
import pytest

from msl.nlf import LinearModel

TMP_DIR = ''


def setup_module():
    global TMP_DIR
    TMP_DIR = tempfile.mkdtemp(prefix='nlf-test-')


def teardown_module():
    shutil.rmtree(TMP_DIR)


def save_corr_array(array, name) -> str:
    f = os.path.join(TMP_DIR, f'CorrCoeffs {name}.txt')
    np.savetxt(f, array)
    return f


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


def test_fit_correlated():
    model = LinearModel()

    assert model.show_warnings is True
    model.set_correlation('y', 'y', value=0.5)

    # check with set_correlation
    with pytest.warns(UserWarning, match='correlations are specified'):
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=False)

    # UserWarning is suppressed
    model.show_warnings = False
    assert model.show_warnings is False
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=False)

    # re-enable
    model.show_warnings = True
    with pytest.warns(UserWarning, match='correlations are specified'):
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=True)

    # remove
    model.remove_correlations()

    # no warnings
    assert model.show_warnings is True
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=False)

    # set directory
    save_corr_array(np.array([[1., 2.], [3., 4.]]), 'Y-Y')
    model.set_correlation_dir(TMP_DIR)
    assert model.show_warnings is True
    with pytest.warns(UserWarning, match='correlations are specified'):
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=True)

    # remove
    model.remove_correlations()

    # no warnings
    assert model.show_warnings is True
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[0, 1], debug=True, correlated=False)
