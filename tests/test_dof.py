from math import isinf

from msl.nlf import LinearModel

x = [1.0, 2.0, 3.0, 4.0]
y = [1.1, 1.9, 3.2, 3.7]
p = [0, 1]
uy = [0.1, 0.2, 0.3, 0.4]


def test_unweighted_uncorrelated() -> None:
    with LinearModel() as model:
        result = model.fit(x, y, params=p)
        assert result.dof == 2

        params = model.create_parameters((("a1", 0, True), ("a2", 1)))
        result = model.fit(x, y, params=params)
        assert result.dof == 3

        params = model.create_parameters((("a1", 0, True), ("a2", 1, True)))
        result = model.fit(x, y, params=params)
        assert result.dof == 4

        result.dof = 7.4
        assert result.dof == 7.4


def test_weighted_uncorrelated() -> None:
    with LinearModel(weighted=True) as model:
        result = model.fit(x, y, params=p, uy=uy)
        assert isinf(result.dof)

        params = model.create_parameters((("a1", 0, True), ("a2", 1)))
        result = model.fit(x, y, params=params, uy=uy)
        assert isinf(result.dof)


def test_unweighted_correlated() -> None:
    with LinearModel(correlated=True) as model:
        model.show_warnings = False
        model.set_correlation("y", "y", 0.5)

        result = model.fit(x, y, params=p, uy=uy)
        assert isinf(result.dof)

        params = model.create_parameters((("a1", 0, True), ("a2", 1)))
        result = model.fit(x, y, params=params, uy=uy)
        assert isinf(result.dof)


def test_weighted_correlated() -> None:
    with LinearModel(weighted=True, correlated=True) as model:
        model.set_correlation("y", "y", 0.5)

        result = model.fit(x, y, params=p, uy=uy)
        assert isinf(result.dof)

        params = model.create_parameters((("a1", 0, True), ("a2", 1)))
        result = model.fit(x, y, params=params, uy=uy)
        assert isinf(result.dof)
