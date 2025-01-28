"""Built-in models."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial.polynomial import polyfit

from .model import Model
from .parameters import InputParameters

if TYPE_CHECKING:
    from typing import Any

    from .types import ArrayLike1D, ArrayLike2D

__all__ = (
    "ConstantModel",
    "ExponentialModel",
    "GaussianModel",
    "LinearModel",
    "PolynomialModel",
    "SineModel",
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
    """A model based on a Gaussian function or a normal distribution."""

    def __init__(self, *, normalized: bool = False, **kwargs: Any) -> None:  # noqa: ANN401
        r"""A model based on a Gaussian function or a normal distribution.

        The non-normalized function is defined as

        $$f(x; a) = a_1 e^{-\frac{1}{2}(\frac{x-a_2}{a_3})^2}$$

        whereas, the normalized function is defined as

        $$f(x; a) = \frac{a_1}{a_3\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-a_2}{a_3})^2}$$

        Args:
            normalized: Whether to use the normalized function.
            **kwargs: All additional keyword arguments are passed to [Model][msl.nlf.model.Model].
        """
        exp = "exp(-0.5*((x-a2)/a3)^2)"
        equation = f"a1/(a3*(2*pi)^0.5)*{exp}" if normalized else f"a1*{exp}"
        self._normalized = normalized
        super().__init__(equation, **kwargs)

        # must define after calling super()
        self._factor = "a1"
        if normalized:
            self._composite_equation = f"1/a3*{exp}"
        else:
            self._composite_equation = exp

    def guess(self, x: ArrayLike1D, y: ArrayLike1D, *, n: int = 3) -> InputParameters:  # type: ignore[override]
        r"""Converts the data to a quadratic and calls the [polyfit][numpy.polynomial.polynomial.polyfit]{:target="_blank"} function.

        Args:
            x: The independent variable (stimulus) data.
            y: The dependent variable (response) data.
            n: Uses the `n` maximum and the `n` minimum values in `y` to determine the region
                where the peak/dip is located.

        Returns:
            Initial guess for the amplitude (area), $\mu$ and $\sigma$ parameters.
        """  # noqa: E501
        x = np.asanyarray(x)
        y = np.asanyarray(y)
        y_min = _mean_min_n(y, n)
        y_max = _mean_max_n(y, n)
        inverted = abs(y_min) > abs(y_max)

        # only use the points near the peak/dip for the polyfit (0.368=1/e)
        indices = y < 0.368 * y_min if inverted else y > 0.368 * y_max

        x, y = x[indices], y[indices]
        ln_y = np.log(np.absolute(y) + 1e-15)
        with warnings.catch_warnings():
            # ignore "RankWarning: The fit may be poorly conditioned"
            warnings.simplefilter("ignore")
            a, b, c = polyfit(x, ln_y, 2)

        sigma = np.sqrt(abs(1 / (2 * c)))
        mu = -b / (2 * c)
        amplitude = np.exp(a - b**2 / (4 * c))
        if inverted:
            amplitude *= -1

        if self._normalized:
            a1_label = "area"
            a1 = amplitude * sigma * math.sqrt(2.0 * math.pi)
        else:
            a1_label = "amplitude"
            a1 = amplitude

        return InputParameters(
            (
                ("a1", a1, False, a1_label),
                ("a2", mu, False, "mu"),
                ("a3", sigma, False, "sigma"),
            )
        )


class LinearModel(Model):
    """A model based on a linear function."""

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        """A model based on a linear function.

        The function is defined as

        $$f(x; a) = a_1 + a_2 x$$

        Args:
            **kwargs: All keyword arguments are passed to [Model][msl.nlf.model.Model].
        """
        a2x = "a2*x"
        super().__init__(f"a1+{a2x}", **kwargs)

        # must define after calling super()
        self._offset = "a1"
        self._composite_equation = a2x

    def guess(self, x: ArrayLike1D, y: ArrayLike1D, **kwargs: Any) -> InputParameters:  # type: ignore[override]  # noqa: ANN401, ARG002
        """Calls the [polyfit][numpy.polynomial.polynomial.polyfit]{:target="_blank"} function.

        Args:
            x: The independent variable (stimulus) data.
            y: The dependent variable (response) data.
            **kwargs: Ignored. No keyword arguments are used.

        Returns:
            Initial guess for the intercept and slope.
        """
        a1, a2 = polyfit(x, y, 1)
        return InputParameters((("a1", a1, False, "intercept"), ("a2", a2, False, "slope")))


class SineModel(Model):
    """A model based on a sine function."""

    def __init__(self, *, angular: bool = False, **kwargs: Any) -> None:  # noqa: ANN401
        r"""A model based on a sine function.

        If `angular` is `False` (default), the function is defined as

        $$f(x; a) = a_1 \sin(2 \pi a_2 x + a_3)$$

        and $a_2$ represents the frequency of oscillation, otherwise

        $$f(x; a) = a_1 \sin(a_2 x + a_3)$$

        and $a_2$ represents the angular frequency.

        Args:
            angular: Whether to use angular frequency in the equation.
            **kwargs: All keyword arguments are passed to [Model][msl.nlf.model.Model].
        """
        sin = "sin(a2*x+a3)" if angular else "sin(2*pi*a2*x+a3)"
        super().__init__(f"a1*{sin}", **kwargs)

        # must define after calling super()
        self._factor = "a1"
        self._composite_equation = sin
        self._angular = angular

    def guess(self, x: ArrayLike1D, y: ArrayLike1D, *, uniform: bool = True, n: int = 11) -> InputParameters:  # type: ignore[override]
        r"""Uses an FFT to determine the amplitude and angular frequency.

        Args:
            x: The independent variable (stimulus) data. The `x` data must be sorted (smallest to largest).
            y: The dependent variable (response) data.
            uniform: Whether the `x` data has uniform spacing between each value.
            n: The number of sub-intervals to break up [0, $2\pi$) to determine the phase guess.

        Returns:
            Initial guess for the amplitude, angular frequency and phase.
        """
        x, y = np.asanyarray(x), np.asanyarray(y)
        dx = abs(x[1] - x[0]) if uniform else np.mean(np.abs(np.diff(x)))
        y -= np.mean(y)
        two_pi = 2 * np.pi

        frequencies = np.fft.fftfreq(len(x), dx)
        amplitudes = np.absolute(np.fft.fft(y))
        argmax = np.argmax(amplitudes)
        a1 = 2.0 * amplitudes[argmax] / len(amplitudes)
        frequency = abs(frequencies[argmax])
        phases = np.linspace(0, two_pi, n, endpoint=False)
        norms = [np.linalg.norm(y - a1 * np.sin(two_pi * frequency * x + p)) for p in phases]
        a3 = phases[np.argmin(norms)]

        if self._angular:
            a2_value, a2_name = two_pi * frequency, "omega"
        else:
            a2_value, a2_name = frequency, "frequency"

        return InputParameters(
            (
                ("a1", a1, False, "amplitude"),
                ("a2", a2_value, False, a2_name),
                ("a3", a3, False, "phase"),
            )
        )


class ExponentialModel(Model):
    """A model based on an exponential function."""

    def __init__(self, *, cumulative: bool = False, **kwargs: Any) -> None:  # noqa: ANN401
        """A model based on an exponential function.

        The non-cumulative function is defined as

        $$f(x; a) = a_1 e^{-a_2 x}$$

        whereas, the cumulative function is defined as

        $$f(x; a) = a_1 (1-e^{-a_2 x})$$

        Args:
            cumulative: Whether to use the cumulative function.
            **kwargs: All keyword arguments are passed to [Model][msl.nlf.model.Model].
        """
        self._cumulative = cumulative
        exp = "(1-exp(-a2*x))" if cumulative else "exp(-a2*x)"
        super().__init__(f"a1*{exp}", **kwargs)

        # must define after calling super()
        self._factor = "a1"
        self._composite_equation = exp

    def guess(self, x: ArrayLike1D, y: ArrayLike1D, *, n: int = 3) -> InputParameters:  # type: ignore[override]
        """Linearizes the equation and calls the [polyfit][numpy.polynomial.polynomial.polyfit]{:target="_blank"} function.

        Args:
            x: The independent variable (stimulus) data.
            y: The dependent variable (response) data.
            n: For a cumulative equation, uses the maximum `n` values in `y` to calculate the mean
                and assigns the mean value as the amplitude guess.

        Returns:
            Initial guess for the amplitude and decay factor.
        """  # noqa: E501
        amplitude = None
        y = np.asanyarray(y)
        if self._cumulative:
            amplitude = _mean_max_n(y, n)
            abs_y = np.absolute(1.0 - y / amplitude)
        else:
            abs_y = np.absolute(y)

        ln_y = np.log(abs_y + 1.0e-15)  # make sure ln(0) is not calculated
        intercept, slope = polyfit(x, ln_y, 1)
        decay = -slope
        if amplitude is None:
            amplitude = np.exp(intercept)

        return InputParameters((("a1", amplitude, False, "amplitude"), ("a2", decay, False, "decay")))


class PolynomialModel(Model):
    """A model based on a polynomial function."""

    def __init__(self, n: int, **kwargs: Any) -> None:  # noqa: ANN401
        r"""A model based on a polynomial function.

        The function is defined as

        $$f(x; a) = \sum_{i=1}^{n} a_i x^{i-1}$$

        Args:
            n: The order of the polynomial (n $\geq$ 1).
            **kwargs: All keyword arguments are passed to [Model][msl.nlf.model.Model].
        """
        if n < 1:
            msg = "Polynomial order must be >= 1"
            raise ValueError(msg)

        eqn = ["a1"]
        for i in range(1, n + 1):
            if i == 1:
                eqn.append("+a2*x")
            else:
                eqn.append(f"+a{i + 1}*x^{i}")

        self._n = n
        equation = "".join(eqn)
        super().__init__(equation, **kwargs)

        # must define after calling super()
        self._offset = "a1"
        self._composite_equation = equation[3:]

    def guess(self, x: ArrayLike1D, y: ArrayLike1D, **kwargs: Any) -> InputParameters:  # type: ignore[override]  # noqa: ANN401, ARG002
        """Calls the [polyfit][numpy.polynomial.polynomial.polyfit]{:target="_blank"} function.

        Args:
            x: The independent variable (stimulus) data.
            y: The dependent variable (response) data.
            **kwargs: Ignored. No keyword arguments are used.

        Returns:
            Initial guess for the polynomial coefficients.
        """
        params = InputParameters()
        for i, coeff in enumerate(polyfit(x, y, self._n), start=1):
            if i == 1:
                label = "a1"
            elif i == 2:  # noqa: PLR2004
                label = "a2*x"
            else:
                label = f"a{i}*x^{i - 1}"
            params[f"a{i}"] = coeff, False, label
        return params


class ConstantModel(Model):
    """A model based on a constant."""

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        r"""A model based on a constant (i.e., a single parameter with no $x$ dependence).

        The function is defined as

        $$f(x; a) = a_1$$

        Args:
            **kwargs: All keyword arguments are passed to [Model][msl.nlf.model.Model].
        """
        super().__init__("a1", **kwargs)

        # must define after calling super()
        self._offset = "a1"
        self._composite_equation = ""

    def guess(self, x: ArrayLike1D | ArrayLike2D, y: ArrayLike1D, *, n: int | None = None) -> InputParameters:  # type: ignore[override]  # noqa: ARG002
        """Calculates the mean value of `y`.

        Args:
            x: The independent variable (stimulus) data. The data is not used.
            y: The dependent variable (response) data.
            n: The number of values in `y` to use to calculate the mean. If not specified,
                all values are used. If a positive integer then the first `n` values are used,
                otherwise, the last `n` values are used.

        Returns:
            Initial guess for the constant.
        """
        if n is None:
            mean = np.mean(y)
        elif n >= 0:
            mean = np.mean(y[:n])
        else:
            mean = np.mean(y[n:])
        return InputParameters([("a1", float(mean), False, "constant")])
