import os
import shutil
import tempfile

import numpy as np
import pytest

from msl.nlf import LinearModel
from msl.nlf import Model
from msl.nlf import load
from msl.nlf import version_info

TMP_DIR = ''


def setup_module():
    global TMP_DIR
    TMP_DIR = tempfile.mkdtemp(prefix='nlf-test-')


def teardown_module():
    shutil.rmtree(TMP_DIR)


def test_must_specify():
    with LinearModel() as model:
        with pytest.raises(ValueError, match=r'x data'):
            model.save('')
        with pytest.raises(ValueError, match=r'y data'):
            model.save('', x=[1, 2])
        with pytest.raises(ValueError, match=r'params'):
            model.save('', x=[1, 2], y=[1, 2])


def test_bad_extension():
    with LinearModel() as model:
        with pytest.raises(ValueError, match=r'\.nlf'):
            model.save('filename.msl', x=[1, 2], y=[3, 4], params=[1, 1])


def test_overwrite():
    path = os.path.join(TMP_DIR, 'overwrite.nlf')
    with LinearModel() as model:
        model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[1, 1])
        model.save(path)
        with pytest.raises(FileExistsError):
            model.save(path)
        model.save(path, overwrite=True)


def test_default_x_y():
    x = [[1000, 2000, 3000]]
    y = [0.1, 0.2, 0.3]
    path = os.path.join(TMP_DIR, 'default_x_y.nlf')
    with LinearModel() as model:
        model.fit(x, y, params=[1, 1])
        model.save(path)

    loaded = load(path)
    assert loaded.num_variables == 1
    assert loaded.num_parameters == 2
    assert loaded.comments == ''
    assert loaded.equation == 'a1+a2*x'
    assert loaded.nlf_path.endswith('x_y.nlf')
    assert loaded.nlf_version == f'{version_info.major}.{version_info.minor}'
    assert np.array_equal(loaded.x, x)
    assert np.array_equal(loaded.y, y)
    assert np.array_equal(loaded.ux, [[0, 0, 0]])
    assert np.array_equal(loaded.uy, [0, 0, 0])


def test_override_x_y():
    path = os.path.join(TMP_DIR, 'override_x_y.nlf')
    with LinearModel() as model:
        model.fit([1000, 2000, 3000], [0.1, 0.2, 0.3])

        model.save(path, x=[1, 2, 3])
        loaded = load(path)
        assert np.array_equal(loaded.x, [[1, 2, 3]])
        assert np.array_equal(loaded.y, [0.1, 0.2, 0.3])
        assert np.array_equal(loaded.ux, [[0, 0, 0]])
        assert np.array_equal(loaded.uy, [0, 0, 0])

        model.save(path, y=[-1, 2, -3], overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.x, [[1, 2, 3]])
        assert np.array_equal(loaded.y, [-1, 2, -3])
        assert np.array_equal(loaded.ux, [[0, 0, 0]])
        assert np.array_equal(loaded.uy, [0, 0, 0])

        model.save(path, y=[11, 22, 33], x=[[101, 202, 303]], overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.x, [[101, 202, 303]])
        assert np.array_equal(loaded.y, [11, 22, 33])
        assert np.array_equal(loaded.ux, [[0, 0, 0]])
        assert np.array_equal(loaded.uy, [0, 0, 0])

        model.fit([1000, 2000, 3000], [0.1, 0.2, 0.3])
        model.save(path, overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.x, [[1000, 2000, 3000]])
        assert np.array_equal(loaded.y, [0.1, 0.2, 0.3])
        assert np.array_equal(loaded.ux, [[0, 0, 0]])
        assert np.array_equal(loaded.uy, [0, 0, 0])


def test_override_params():
    x = [1, 2, 3]
    y = [0.1, 0.2, 0.3]
    path = os.path.join(TMP_DIR, 'override_params.nlf')
    with LinearModel() as model:
        # use guess
        model.fit(x, y)
        model.save(path)
        loaded = load(path)
        a, b = loaded.params.values()
        assert pytest.approx(a, abs=1e-15) == 0
        assert pytest.approx(b, abs=1e-15) == 0.1
        assert np.array_equal(loaded.params.constants(), [False, False])
        assert loaded.params.names() == ['a1', 'a2']
        assert loaded.params.labels() == [None, None]

        # define as array
        model.fit(x, y, params=[-1, 2])
        model.save(path, overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.params.values(), [-1, 2])
        assert np.array_equal(loaded.params.constants(), [False, False])
        assert loaded.params.names() == ['a1', 'a2']
        assert loaded.params.labels() == [None, None]

        # don't call fit before save
        model.save(path, params=[1, 2], overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.params.values(), [1, 2])
        assert np.array_equal(loaded.params.constants(), [False, False])
        assert loaded.params.names() == ['a1', 'a2']
        assert loaded.params.labels() == [None, None]

        # params not specified, so get the previous result
        model.save(path, overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.params.values(), [1, 2])
        assert np.array_equal(loaded.params.constants(), [False, False])
        assert loaded.params.names() == ['a1', 'a2']
        assert loaded.params.labels() == [None, None]

        # create InputParameters instance
        params = model.create_parameters((
            ('a1', -5.5, True, 'intercept'),
            ('a2', 10.1, False, 'slope'),
        ))
        model.save(path, params=params, overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.params.values(), [-5.5, 10.1])
        assert np.array_equal(loaded.params.constants(), [True, False])
        assert loaded.params.names() == ['a1', 'a2']

        # since the Delphi GUI does not use labels, the labels
        # are not preserved during the save-load procedure
        assert loaded.params.labels() == [None, None]


def test_override_ux_uy():
    x = [1, 2, 3]
    y = [0.1, 0.2, 0.3]
    path = os.path.join(TMP_DIR, 'override_ux_uy.nlf')
    with LinearModel() as model:
        model.show_warnings = False  # suppress weighted fit warnings

        # all zeros
        model.fit(x, y)
        model.save(path)
        loaded = load(path)
        assert np.array_equal(loaded.x, [x])
        assert np.array_equal(loaded.y, y)
        assert np.array_equal(loaded.ux, [[0, 0, 0]])
        assert np.array_equal(loaded.uy, [0, 0, 0])

        # ux changes
        model.save(path, ux=[0.01, 0.02, 0.03], overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.x, [x])
        assert np.array_equal(loaded.y, y)
        assert np.array_equal(loaded.ux, [[0.01, 0.02, 0.03]])
        assert np.array_equal(loaded.uy, [0, 0, 0])

        # ux same as previous call, uy changes
        model.save(path, uy=[0.3, 0.1, 0.2], overwrite=True)
        loaded = load(path)
        assert np.array_equal(loaded.x, [x])
        assert np.array_equal(loaded.y, y)
        assert np.array_equal(loaded.ux, [[0.01, 0.02, 0.03]])
        assert np.array_equal(loaded.uy, [0.3, 0.1, 0.2])


@pytest.mark.parametrize(
    'comments',
    ['', 'hello', 'hello\nworld', '{hello}\n{world}', 'the data=from today'])
def test_comments(comments):
    path = os.path.join(TMP_DIR, 'comments.nlf')
    with LinearModel() as model:
        model.fit([1, 2, 3], [0.1, 0.2, 0.3])
        model.save(path, comments=comments, overwrite=True)
        loaded = load(path)
        assert loaded.comments == comments


def test_save_demo():
    path = os.path.join(TMP_DIR, 'demo.nlf')
    x = np.array([[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]])
    y = np.array([1.1, 1.9, 3.2, 3.7])
    a = np.array([0, 0.9, 0])
    uy = np.array([0.5, 0.5, 0.5, 0.5])
    ux = np.array([[0.01, 0.02, 0.03, 0.04], [0.002, 0.004, 0.006, 0.008]])
    with Model('a1+a2*(x+exp(a3*x))+x2', weighted=True, correlated=True) as model:
        model.set_correlation('y', 'y', value=0.5)
        model.set_correlation('x', 'x', value=0.8)
        model.options(second_derivs_B=False, max_iterations=200)
        result1 = model.fit(x, y, params=a, uy=uy, ux=ux, tolerance=1.23e-12, delta=0.01)
        model.save(path, comments='demo')

    loaded = load(path)
    assert loaded.num_variables == 2
    assert loaded.num_parameters == 3
    assert loaded.comments == 'demo'
    assert loaded.equation == 'a1+a2*(x+exp(a3*x))+x2'
    assert loaded.nlf_path.endswith('demo.nlf')
    assert loaded.nlf_version == f'{version_info.major}.{version_info.minor}'
    assert np.array_equal(loaded.x, x)
    assert np.array_equal(loaded.y, y)
    assert np.array_equal(loaded.ux, ux)
    assert np.array_equal(loaded.uy, uy)
    assert np.array_equal(loaded.params.values(), a)
    assert np.array_equal(loaded.params.constants(), [False, False, False])
    assert loaded.params.names() == ['a1', 'a2', 'a3']
    assert loaded.params.labels() == [None, None, None]

    data = loaded.fit(loaded.x, loaded.y, params=loaded.params, ux=loaded.ux, uy=loaded.uy, debug=True)

    assert data.correlated is True
    assert data.delta == 0.01
    assert data.equation == 'a1+a2*(x+exp(a3*x))+x2'
    assert data.fit_method == model.FitMethod.LM
    assert data.max_iterations == 200
    assert np.array_equal(data.params.values(), a)
    assert np.array_equal(data.params.constants(), [False, False, False])
    assert data.params.names() == ['a1', 'a2', 'a3']
    assert data.params.labels() == [None, None, None]
    assert data.second_derivs_B is False
    assert data.second_derivs_H is True
    assert data.tolerance == 1.23e-12
    assert np.array_equal(data.ux, ux)
    assert np.array_equal(data.uy, uy)
    assert data.uy_weights_only is False
    assert data.weighted is True
    assert np.array_equal(data.x, x)
    assert np.array_equal(data.y, y)

    c = data.correlations
    assert np.array_equal(c.is_correlated, [[True, False, False],
                                            [False, True, False],
                                            [False, False, False]])
    assert c.data[0].path.endswith('X1-X1.txt')
    assert c.data[1].path.endswith('Y-Y.txt')
    assert np.array_equal(c.data[0].coefficients, np.array([[1.0, 0.8, 0.8, 0.8],
                                                            [0.8, 1.0, 0.8, 0.8],
                                                            [0.8, 0.8, 1.0, 0.8],
                                                            [0.8, 0.8, 0.8, 1.0]]))
    assert np.array_equal(c.data[1].coefficients, np.array([[1.0, 0.5, 0.5, 0.5],
                                                            [0.5, 1.0, 0.5, 0.5],
                                                            [0.5, 0.5, 1.0, 0.5],
                                                            [0.5, 0.5, 0.5, 1.0]]))

    result2 = loaded.fit(loaded.x, loaded.y, params=loaded.params, ux=loaded.ux, uy=loaded.uy)
    assert np.array_equal(result1.params.values(), result2.params.values())
    assert np.array_equal(result1.params.uncerts(), result2.params.uncerts())
