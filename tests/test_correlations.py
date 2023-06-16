import os
import shutil
import tempfile

import numpy as np
import pytest

from msl.nlf import *

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


def test_set_correlation_dir_raises():
    # the Model type is irrelevant
    with LinearModel() as model:
        with pytest.raises(OSError, match=r'not a valid directory'):
            model.set_correlation_dir('does-not-exist')

        # empty string valid
        model.set_correlation_dir('')

        # must still be a string, even when call with None
        model.set_correlation_dir(None)
        assert model._corr_dir == ''


def test_set_correlation_dir():
    dummy1 = np.array([[1., 2.], [3., 4.]])
    dummy2 = np.array([[5., 6.], [7., 8.]])
    dummy3 = np.array([[-1., -2.], [-3., -4.]])
    kwargs = dict(x=[], y=[], params=[1, 2], debug=True)

    # the Model type is irrelevant
    with LinearModel() as model:
        model.show_warnings = False

        # no correlation files exist
        model.set_correlation_dir(os.path.dirname(__file__))
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 0
            assert np.array_equal(c.is_correlated, [[False, False], [False, False]])

        model.set_correlation_dir(TMP_DIR)

        # The correlations are independent of whether
        # fit(correlated=False) or fit(correlated=True)
        #
        # The correlated flag is used by the DLL to decide if
        # correlations are to be used

        # create Y-Y correlation file
        y_y_file = save_corr_array(dummy1, 'Y-Y')
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 1
            assert c.data[0].path == y_y_file
            assert np.array_equal(c.data[0].coefficients, dummy1)
            assert np.array_equal(c.is_correlated, [[True, False], [False, False]])

        os.remove(y_y_file)

        # create Y-X1 correlation file
        y_x1_file = save_corr_array(dummy2, 'Y-X1')
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 1
            assert c.data[0].path == y_x1_file
            assert np.array_equal(c.data[0].coefficients, dummy2)
            assert np.array_equal(c.is_correlated, [[False, True], [True, False]])

        # create X1-X1 correlation file (Y-X1 still exists)
        x1_x1_file = save_corr_array(dummy3, 'X1-X1')
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 2
            assert c.data[0].path == x1_x1_file
            assert np.array_equal(c.data[0].coefficients, dummy3)
            assert c.data[1].path == y_x1_file
            assert np.array_equal(c.data[1].coefficients, dummy2)
            assert np.array_equal(c.is_correlated, [[False, True], [True, True]])

        # create Y-Y correlation file (X1-X1 and Y-X1 still exist)
        y_y_file = save_corr_array(dummy1, 'Y-Y')
        model.set_correlation_dir(TMP_DIR)
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 3
            assert c.data[0].path == x1_x1_file
            assert np.array_equal(c.data[0].coefficients, dummy3)
            assert c.data[1].path == y_x1_file
            assert np.array_equal(c.data[1].coefficients, dummy2)
            assert c.data[2].path == y_y_file
            assert np.array_equal(c.data[2].coefficients, dummy1)
            assert np.array_equal(c.is_correlated, [[True, True], [True, True]])

        # pretend that no correlation files exist
        model.set_correlation_dir(None)
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 0
            assert np.array_equal(c.is_correlated, [[False, False], [False, False]])


def test_set_correlation_raises():
    # the Model type is irrelevant
    with LinearModel() as model:

        with pytest.raises(ValueError, match=r"either 'value' or 'matrix'"):
            model.set_correlation('y', 'y')

        with pytest.raises(ValueError, match=r"both 'value' and 'matrix'"):
            model.set_correlation('y', 'y', matrix=[[1, 1], [1, 1]], value=1)

        for name in ['a1', ' ', 'y1']:
            with pytest.raises(ValueError, match=r'Invalid correlation variable name'):
                model.set_correlation(name, 'y', value=1)

        for name in ['x0', 'x3']:
            with pytest.raises(ValueError, match=r'X index outside of range'):
                model.set_correlation('y', name, value=1)

        for m in [[1, 2], 3, np.empty((3, 4, 5))]:
            with pytest.raises(ValueError, match=r'Invalid correlation matrix dimension'):
                model.set_correlation('y', 'y', matrix=m)


def test_set_correlation():
    x = np.array([[1., 2., 3., 4.], [0.1, 0.2, 0.3, 0.4]])
    y = np.array([1.1, 1.9, 3.2, 3.7])
    p = np.array([0., 0.9, 0.])
    kwargs = dict(x=x, y=y, params=p, debug=True)

    # the Model type is irrelevant
    with Model('a1+a2*(x+exp(a3*x))+x2') as model:
        model.show_warnings = False

        # no correlations exist
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 0
            assert np.array_equal(c.is_correlated, [[False, False, False],
                                                    [False, False, False],
                                                    [False, False, False]])

        # The correlations are independent of whether
        # fit(correlated=False) or fit(correlated=True)
        #
        # The correlated flag is used by the DLL to decide if
        # correlations are to be used

        # create Y-Y correlation file
        model.set_correlation('y', 'y', value=1)
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 1
            assert c.data[0].path == os.path.join(model._corr_dir, 'CorrCoeffs Y-Y.txt')
            assert np.array_equal(c.data[0].coefficients, np.ones((4, 4)))
            assert np.array_equal(c.is_correlated, [[True, False, False],
                                                    [False, False, False],
                                                    [False, False, False]])

        model.remove_correlations()

        # create Y-X1 correlation file
        # 'x' gets automatically renamed to 'X1'
        # Y-Y correlations were removed
        y_x1_matrix = 7 * np.ones((4, 4))
        model.set_correlation('y', 'x', matrix=y_x1_matrix)
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 1
            assert c.data[0].path == os.path.join(model._corr_dir, 'CorrCoeffs Y-X1.txt')
            assert np.array_equal(c.data[0].coefficients, y_x1_matrix)
            assert np.array_equal(c.is_correlated, [[False, True, False],
                                                    [True, False, False],
                                                    [False, False, False]])

        # create bad X1-X1 correlation file, must have shape (4, 4)
        model.set_correlation('x', 'x', matrix=np.ones((5, 5)))
        with pytest.raises(ValueError, match=r"Invalid 'X1-X1' correlation array shape"):
            model.fit(correlated=boolean, **kwargs)

        # create proper X1-X1 correlation file (Y-X1 still exists)
        x1_x1_matrix = 2*np.ones((4, 4))
        model.set_correlation('x', 'x', matrix=x1_x1_matrix)
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 2
            assert c.data[0].path == os.path.join(model._corr_dir, 'CorrCoeffs X1-X1.txt')
            assert np.array_equal(c.data[0].coefficients, x1_x1_matrix)
            assert c.data[1].path == os.path.join(model._corr_dir, 'CorrCoeffs Y-X1.txt')
            assert np.array_equal(c.data[1].coefficients, y_x1_matrix)
            assert np.array_equal(c.is_correlated, [[False, True, False],
                                                    [True, True, False],
                                                    [False, False, False]])

        # create Y-Y correlation file (X1-X1 and Y-X1 still exist)
        model.set_correlation('y', 'y', value=3)
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 3
            assert c.data[0].path == os.path.join(model._corr_dir, 'CorrCoeffs X1-X1.txt')
            assert np.array_equal(c.data[0].coefficients, x1_x1_matrix)
            assert c.data[1].path == os.path.join(model._corr_dir, 'CorrCoeffs Y-X1.txt')
            assert np.array_equal(c.data[1].coefficients, y_x1_matrix)
            assert c.data[2].path == os.path.join(model._corr_dir, 'CorrCoeffs Y-Y.txt')
            assert np.array_equal(c.data[2].coefficients, np.array([[1., 3., 3., 3.],
                                                                    [3., 1., 3., 3.],
                                                                    [3., 3., 1., 3.],
                                                                    [3., 3., 3., 1.]]))
            assert np.array_equal(c.is_correlated, [[True, True, False],
                                                    [True, True, False],
                                                    [False, False, False]])

        # create X2-Y correlation file (Y-Y, X1-X1 and Y-X1 still exist)
        model.set_correlation('x2', 'y', value=5)
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 4
            assert c.data[0].path == os.path.join(model._corr_dir, 'CorrCoeffs X1-X1.txt')
            assert np.array_equal(c.data[0].coefficients, x1_x1_matrix)
            assert c.data[1].path == os.path.join(model._corr_dir, 'CorrCoeffs X2-Y.txt')
            assert np.array_equal(c.data[1].coefficients, np.array([[1., 5., 5., 5.],
                                                                    [5., 1., 5., 5.],
                                                                    [5., 5., 1., 5.],
                                                                    [5., 5., 5., 1.]]))
            assert c.data[2].path == os.path.join(model._corr_dir, 'CorrCoeffs Y-X1.txt')
            assert np.array_equal(c.data[2].coefficients, y_x1_matrix)
            assert c.data[3].path == os.path.join(model._corr_dir, 'CorrCoeffs Y-Y.txt')
            assert np.array_equal(c.data[3].coefficients, np.array([[1., 3., 3., 3.],
                                                                    [3., 1., 3., 3.],
                                                                    [3., 3., 1., 3.],
                                                                    [3., 3., 3., 1.]]))
            assert np.array_equal(c.is_correlated, [[True, True, True],
                                                    [True, True, False],
                                                    [True, False, False]])

        # again, no correlation files exist
        model.remove_correlations()
        for boolean in (False, True):
            data = model.fit(correlated=boolean, **kwargs)
            c = data.correlations
            assert len(c.data) == 0
            assert np.array_equal(c.is_correlated, [[False, False, False],
                                                    [False, False, False],
                                                    [False, False, False]])
