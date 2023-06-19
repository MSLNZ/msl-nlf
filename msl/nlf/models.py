"""
Predefined models

* :class:`.ConstantModel`
* :class:`.CosineModel`
* :class:`.ExponentialModel`
* :class:`.GaussianModel`
* :class:`.LinearModel`
* :class:`.PolynomialModel`
* :class:`.SineModel`
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.polynomial.polynomial import polyfit

from .model import ArrayLike1D
from .model import Model
from .parameter import InputParameters

__all__ = (
    'GaussianModel', 'ExponentialModel', 'SineModel', 'CosineModel',
    'ConstantModel', 'LinearModel', 'PolynomialModel',
)


def _mean_max_n(array: ArrayLike1D, n: int) -> float:
    """Return the mean of the maximum *n* values in *array*."""
    a = np.asanyarray(array)
    indices = np.argpartition(a, -n)[-n:]
    return float(np.mean(a[indices]))


def _mean_min_n(array: ArrayLike1D, n: int) -> float:
    """Return the mean of the minimum *n* values in *array*."""
    a = np.asanyarray(array)
    indices = np.argpartition(a, n)[:n]
    return float(np.mean(a[indices]))


class GaussianModel(Model):

    def __init__(self, normalized: bool = False, **kwargs) -> None:
        r"""A model based on a Gaussian function or a normal distribution.

        The non-normalized function is defined as

        .. math::

            f(x; a) = a_1 e^{-\frac{1}{2}(\frac{x-a_2}{a_3})^2}

        whereas, the normalized function is defined as

        .. math::

            f(x; a) = \frac{a_1}{a_3\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-a_2}{a_3})^2}

        Parameters
        ----------
        normalized
            Whether to use the normalized function.
        **kwargs
            All additional keyword arguments are passed to :class:`~msl.nlf.model.Model`.
        """
        exp = 'exp(-0.5*((x-a2)/a3)^2)'
        if normalized:
            sq2pi = math.sqrt(2.0 * math.pi)
            equation = f'a1/(a3*{sq2pi})*{exp}'
        else:
            equation = f'a1*{exp}'
        self._normalized = normalized
        super().__init__(equation, **kwargs)

        # must define after calling super()
        self._factor = 'a1'
        if normalized:
            self._composite_equation = f'1/a3*{exp}'
        else:
            self._composite_equation = exp

    def guess(self,
              x: ArrayLike1D,
              y: ArrayLike1D,
              n: int = 3) -> InputParameters:
        """Converts the data to a quadratic and calls the
        :func:`~numpy.polynomial.polynomial.polyfit` function.

        Parameters
        ----------
        x
            The independent variable (stimulus) data.
        y
            The dependent variable (response) data.
        n
            Uses the *n* maximum and the *n* minimum values in *y* to
            determine the region where the peak/dip is located.

        Returns
        -------
        :class:`.InputParameters`
            Initial guess for the amplitude (area), :math:`\\mu` and
            :math:`\\sigma` parameters.
        """
        y = np.asanyarray(y)
        y_min = _mean_min_n(y, n)
        y_max = _mean_max_n(y, n)
        inverted = abs(y_min) > abs(y_max)

        # only use the points near the peak/dip for the polyfit
        if inverted:
            indices = y < 0.368 * y_min  # 0.368=1/e
        else:
            indices = y > 0.368 * y_max

        x, y = x[indices], y[indices]
        ln_y = np.log(np.absolute(y) + 1e-15)
        with warnings.catch_warnings():
            # ignore "RankWarning: The fit may be poorly conditioned"
            warnings.simplefilter('ignore')
            a, b, c = polyfit(x, ln_y, 2)  # noqa

        sigma = np.sqrt(abs(1/(2*c)))
        mu = -b / (2*c)
        amplitude = np.exp(a - b**2/(4*c))
        if inverted:
            amplitude *= -1

        if self._normalized:
            a1_label = 'area'
            a1 = amplitude * sigma * math.sqrt(2.0 * math.pi)
        else:
            a1_label = 'amplitude'
            a1 = amplitude

        return InputParameters((
            ('a1', a1, False, a1_label),
            ('a2', mu, False, 'mu'),
            ('a3', sigma, False, 'sigma'),))


class LinearModel(Model):

    def __init__(self, **kwargs) -> None:
        """A model based on a linear function.

        The function is defined as

        .. math::

            f(x; a) = a_1 + a_2 x

        Parameters
        ----------
        **kwargs
            All keyword arguments are passed to :class:`~msl.nlf.model.Model`.
        """
        a2x = 'a2*x'
        super().__init__(f'a1+{a2x}', **kwargs)

        # must define after calling super()
        self._offset = 'a1'
        self._composite_equation = a2x

    def guess(self, x: ArrayLike1D, y: ArrayLike1D, **kwargs) -> InputParameters:
        """Calls the :func:`~numpy.polynomial.polynomial.polyfit` function.

        Parameters
        ----------
        x
            The independent variable (stimulus) data.
        y
            The dependent variable (response) data.
        **kwargs
            No keyword arguments are used.

        Returns
        -------
        :class:`.InputParameters`
            Initial guess for the slope and intercept.
        """
        a1, a2 = polyfit(x, y, 1)
        return InputParameters((('a1', a1, False, 'intercept'),
                                ('a2', a2, False, 'slope')))


class SineModel(Model):

    def __init__(self, **kwargs) -> None:
        """A model based on a sine function.

        The function is defined as

        .. math::

            f(x; a) = a_1 sin(a_2 x + a_3)

        Parameters
        ----------
        **kwargs
            All keyword arguments are passed to :class:`~msl.nlf.model.Model`.
        """
        sin = 'sin(a2*x+a3)'
        super().__init__(f'a1*{sin}', **kwargs)

        # must define after calling super()
        self._factor = 'a1'
        self._composite_equation = sin


class CosineModel(Model):

    def __init__(self, **kwargs) -> None:
        """A model based on a cosine function.

        The function is defined as

        .. math::

            f(x; a) = a_1 cos(a_2 x + a_3)

        Parameters
        ----------
        **kwargs
            All keyword arguments are passed to :class:`~msl.nlf.model.Model`.
        """
        cos = 'cos(a2*x+a3)'
        super().__init__(f'a1*{cos}', **kwargs)

        # must define after calling super()
        self._factor = 'a1'
        self._composite_equation = cos


class ExponentialModel(Model):

    def __init__(self, cumulative: bool = False, **kwargs) -> None:
        """A model based on an exponential function.

        The non-cumulative function is defined as

        .. math::

            f(x; a) = a_1 e^{-a_2 x}

        whereas, the cumulative function is defined as

        .. math::

            f(x; a) = a_1 (1-e^{-a_2 x})

        Parameters
        ----------
        cumulative
            Whether to use the cumulative function.
        **kwargs
            All keyword arguments are passed to :class:`~msl.nlf.model.Model`.
        """
        self._cumulative = cumulative
        exp = '(1-exp(-a2*x))' if cumulative else 'exp(-a2*x)'
        super().__init__(f'a1*{exp}', **kwargs)

        # must define after calling super()
        self._factor = 'a1'
        self._composite_equation = exp

    def guess(self,
              x: ArrayLike1D,
              y: ArrayLike1D,
              n: int = 3) -> InputParameters:
        """Linearizes the equation and calls the
        :func:`~numpy.polynomial.polynomial.polyfit` function.

        Parameters
        ----------
        x
            The independent variable (stimulus) data.
        y
            The dependent variable (response) data.
        n
            For a cumulative equation, finds the maximum *n*
            values in *y* and calculates the mean.

        Returns
        -------
        :class:`.InputParameters`
            Initial guess for the amplitude and decay factor.
        """
        amplitude = None
        y = np.asanyarray(y)
        if self._cumulative:
            amplitude = _mean_max_n(y, n)
            abs_y = np.absolute(1.0-y/amplitude)
        else:
            abs_y = np.absolute(y)

        ln_y = np.log(abs_y + 1.e-15)  # make sure ln(0) is not calculated
        intercept, slope = polyfit(x, ln_y, 1)
        decay = -slope
        if amplitude is None:
            amplitude = np.exp(intercept)

        return InputParameters((('a1', amplitude, False, 'amplitude'),
                                ('a2', decay, False, 'decay')))


class PolynomialModel(Model):

    def __init__(self, n: int, **kwargs) -> None:
        r"""A model based on a polynomial function.

        The function is defined as

        .. math::

            f(x; a) = \sum_{i=1}^{n} a_i x^{i-1}

        Parameters
        ----------
        n
            The order of the polynomial (n :math:`\geq` 1).
        **kwargs
            All keyword arguments are passed to :class:`~msl.nlf.model.Model`.
        """
        if n < 1:
            raise ValueError('Polynomial order must be >= 1')

        eqn = ['a1']
        for i in range(1, n+1):
            if i == 1:
                eqn.append('+a2*x')
            else:
                eqn.append(f'+a{i+1}*x^{i}')

        self._n = n
        equation = ''.join(eqn)
        super().__init__(equation, **kwargs)

        # must define after calling super()
        self._offset = 'a1'
        self._composite_equation = equation[3:]

    def guess(self, x: ArrayLike1D, y: ArrayLike1D, **kwargs) -> InputParameters:
        """Calls the :func:`~numpy.polynomial.polynomial.polyfit` function.

        Parameters
        ----------
        x
            The independent variable (stimulus) data.
        y
            The dependent variable (response) data.
        **kwargs
            No keyword arguments are used.

        Returns
        -------
        :class:`.InputParameters`
            Initial guess for the polynomial coefficients.
        """
        params = InputParameters()
        for i, coeff in enumerate(polyfit(x, y, self._n), start=1):
            if i == 1:
                label = 'a1'
            elif i == 2:
                label = 'a2*x'
            else:
                label = f'a{i}*x^{i-1}'
            params[f'a{i}'] = coeff, False, label
        return params


class ConstantModel(Model):

    def __init__(self, **kwargs) -> None:
        r"""A model based on a constant (i.e., a single parameter with no
        :math:`x` dependence).

        The function is defined as

        .. math::

            f(x; a) = a_1

        Parameters
        ----------
        **kwargs
            All keyword arguments are passed to :class:`~msl.nlf.model.Model`.
        """
        super().__init__('a1', **kwargs)

        # must define after calling super()
        self._offset = 'a1'
        self._composite_equation = ''

    def guess(self, x: ArrayLike1D, y: ArrayLike1D, **kwargs) -> InputParameters:
        """Calculates the mean value of *y*.

        Parameters
        ----------
        x
            The independent variable (stimulus) data. The data is not used.
        y
            The dependent variable (response) data.
        **kwargs
            No keyword arguments are used.

        Returns
        -------
        :class:`.InputParameters`
            Initial guess for the constant.
        """
        return InputParameters([('a1', float(np.mean(y)), False, 'constant')])
