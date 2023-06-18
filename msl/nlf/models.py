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
import math

import numpy as np
from numpy.polynomial.polynomial import polyfit

from .model import ArrayLike1D
from .model import Model
from .parameter import InputParameters

__all__ = (
    'GaussianModel', 'ExponentialModel', 'SineModel', 'CosineModel',
    'ConstantModel', 'LinearModel', 'PolynomialModel',
)


def _max_n(array: ArrayLike1D, n: int) -> np.ndarray:
    """Return the maximum *n* values in *array*."""
    a = np.asarray(array)
    indices = np.argpartition(a, -n)[-n:]
    return a[indices]


def _min_n(array: ArrayLike1D, n: int) -> np.ndarray:
    """Return the minimum *n* values in *array*."""
    a = np.asarray(array)
    indices = np.argpartition(a, n)[:n]
    return a[indices]


class GaussianModel(Model):

    def __init__(self, normalized: bool = False, **kwargs) -> None:
        r"""A model based on a Gaussian function or a normal distribution.

        The un-normalized function is defined as

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
              n: int = 3,
              **kwargs) -> InputParameters:
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
        kwargs
            All other keyword arguments are ignored.

        Returns
        -------
        :class:`.InputParameters`
            Initial guess for the amplitude and decay factor.
        """
        amplitude = None
        y = np.asarray(y)

        if self._cumulative:
            amplitude = np.mean(_max_n(y, n))
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
