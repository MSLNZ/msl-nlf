"""
The word "demo" in this test script filename refers to the example Python script
that P. Saunders wrote to illustrate how to interact with the DLL. This test
verifies that the same results are obtained.
"""
import math

import numpy as np
import pytest

from msl.nlf import Model
from msl.nlf.model import IS_PYTHON_64BIT

if IS_PYTHON_64BIT:
    dlls = ['nlf32', 'nlf64']
else:
    dlls = ['nlf32']


@pytest.mark.parametrize('dll', dlls)
def test_demo(dll):
    x = np.array([[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]])
    y = np.array([1.1, 1.9, 3.2, 3.7])
    a = np.array([0, 0.9, 0])
    sigy = np.array([0.5, 0.5, 0.5, 0.5])
    sigx = np.array([[0.01, 0.02, 0.03, 0.04], [0.002, 0.004, 0.006, 0.008]])

    with Model('a1+a2*(x+exp(a3*x))+x2', dll=dll, weighted=True, correlated=True) as model:
        model.set_correlation('y', 'y', value=0.5)
        model.set_correlation('x', 'x', value=0.8)
        r = model.fit(x=x, y=y, params=a, uy=sigy, ux=sigx)
        assert len(r.params) == 3
        assert r.params['a1'].name == 'a1'
        assert r.params['a2'].name == 'a2'
        assert r.params['a3'].name == 'a3'
        assert r.params['a1'].value == pytest.approx(-0.6101880747640294)
        assert r.params['a2'].value == pytest.approx(0.8100288869777268)
        assert r.params['a3'].value == pytest.approx(4.585005881907852e-05)
        assert r.params['a1'].uncert == pytest.approx(0.6803365385456976)
        assert r.params['a2'].uncert == pytest.approx(0.1584129274256673)
        assert r.params['a3'].uncert == pytest.approx(0.00011904966869376515)
        assert r.params['a1'].label is None
        assert r.params['a2'].label is None
        assert r.params['a3'].label is None
        assert np.allclose(r.covariance,
                           np.array([[0.46285780568034157, -0.08766528257701835, 2.7536838781747962e-05],
                                     [-0.08766528257701835, 0.025094655575569728, -7.88442936685828e-06],
                                     [2.7536838781747962e-05, -7.88442936685828e-06, 1.4172823616095245e-08]]))
        assert np.allclose(r.correlation,
                           np.array([[1.0, -0.81341695617204, 0.33998683044214156],
                                     [-0.81341695617204, 1.0, -0.4180723600492553],
                                     [0.33998683044214156, -0.4180723600492553, 1.0]]))
        assert r.chisq == pytest.approx(0.854875600205648)
        assert r.eof == pytest.approx(0.32710857899179385)
        assert math.isinf(r.dof)
        assert r.iterations == 33
        assert r.calls == 3

        with pytest.raises(PermissionError, match=r'Cannot change ResultParameter name'):
            r.params['a1'].name = 'a9'
        with pytest.raises(PermissionError, match=r'Cannot change ResultParameter value'):
            r.params['a1'].value = 1
        with pytest.raises(PermissionError, match=r'Cannot change ResultParameter uncertainty'):
            r.params['a1'].uncert = 1

        r.params['a1'].label = 'new'
        assert r.params['a1'].label == 'new'
