"""
Various data classes and enums.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    from GTC import multiple_ureal
    from GTC import set_correlation
    from GTC.lib import UncertainReal
except ImportError:
    multiple_ureal = None
    set_correlation = None
    UncertainReal = object

from .parameter import InputParameters
from .parameter import ResultParameters


class FitMethod(Enum):
    """Fitting methods.

    Least squares (LS) minimises the sum of the squares of the vertical distances
    between each point and the fitted curve. The algorithm that is implemented
    for Levenberg Marquardt, Amoeba and Powell is described in
    `Numerical Recipes <http://numerical.recipes/>`_.

    Minimum distance (MD) minimises the sum of the distances (in two dimensions)
    between each point and the fitted curve. This type of fit is not available
    as a correlated fit.

    MiniMax (MM) minimises the value of the maximum absolute y-residual. This
    type of fit is not available as a correlated fit.
    """
    LM = 'Levenberg-Marquardt'
    AMOEBA_LS = 'Amoeba least squares'
    AMOEBA_MD = 'Amoeba minimum distance'
    AMOEBA_MM = 'Amoeba minimax'
    POWELL_LS = 'Powell least squares'
    POWELL_MD = 'Powell minimum distance'
    POWELL_MM = 'Powell minimax'


@dataclass(eq=False, order=False, frozen=True)
class Correlation:
    """Information about correlation coefficients."""

    path: str
    """The path to the correlation file."""

    coefficients: np.ndarray[float]
    """The correlation coefficients."""

    def __repr__(self) -> str:
        # add indentation to the numpy array
        coeff_str = np.array2string(self.coefficients, prefix=' ' * 15)

        return f'Correlation(\n' \
               f'  coefficients={coeff_str}\n' \
               f'  path={self.path!r}\n' \
               f')'


@dataclass(eq=False, order=False, frozen=True)
class Correlations:
    """Information about the correlations for a fit model."""

    data: list[Correlation]
    """A :class:`list` of :class:`.Correlation` objects."""

    is_correlated: np.ndarray[bool]
    """Indicates which variables are correlated. The index 0 corresponds
    to the `y`-variable, the index 1 to `x1`, 2 to `x2`, etc."""

    def __repr__(self) -> str:
        # add indentation to the data
        data_str = ''
        if len(self.data) > 0:
            indent = ' ' * 4
            lines = [indent]
            for item in self.data:
                lines.extend(str(item).splitlines())
                lines[-1] += ','
            lines[-1] = ')'
            data_str = f'\n{indent}'.join(lines)

        # add indentation to the numpy array
        corr_str = np.array2string(self.is_correlated, prefix=' ' * 16)

        return f'Correlations(\n' \
               f'  data=[{data_str}]\n' \
               f'  is_correlated={corr_str}\n' \
               f')'


@dataclass(eq=False, order=False, frozen=True)
class Input:
    """The input data to a fit model."""

    correlated: bool
    """Whether correlations are applied in the fitting process."""

    correlations: Correlations
    """The information about the correlation coefficients."""

    delta: float
    """Only used for Amoeba fitting."""

    equation: str
    """The equation of the fit model."""

    fit_method: FitMethod
    """The method that is used for the fit."""

    max_iterations: int
    """The maximum number of fit iterations allowed."""

    params: InputParameters
    """The input parameters to the fit model."""

    second_derivs_B: bool
    """Whether the second derivatives in the **B** matrix are included in
    the propagation of uncertainty calculations."""

    second_derivs_H: bool
    """Whether the second derivatives in the curvature matrix, **H** (Hessian),
    are included in the propagation of uncertainty calculations."""

    tolerance: float
    """The tolerance value to stop the fitting process."""

    ux: np.ndarray[float]
    """Standard uncertainties in the x (stimulus) data."""

    uy: np.ndarray[float]
    """Standard uncertainties in the y (response) data."""

    uy_weights_only: bool
    """Whether the y uncertainties only or a combination of the x and y
    uncertainties are used to calculate the weights for a weighted fit."""

    weighted: bool
    """Whether to include the standard uncertainties in the fitting
    process to perform a weighted fit."""

    x: np.ndarray[float]
    """The independent variable(s) (stimulus) data."""

    y: np.ndarray[float]
    """The dependent variable (response) data."""

    def __repr__(self) -> str:
        indent = ' ' * 4

        # add indentation to the correlations
        corr = [indent]
        corr.extend(str(self.correlations).splitlines())
        corr[-1] = ')'
        corr_str = f'\n{indent}'.join(corr)

        # add indentation to the parameters
        if not self.params:
            param_str = 'InputParameters()'
        else:
            params = [indent]
            params.extend(str(self.params).splitlines())
            params[-1] = ')'
            param_str = f'\n{indent}'.join(params)

        return f'Input(\n' \
               f'  correlated={self.correlated}\n' \
               f'  correlations={corr_str}\n' \
               f'  delta={self.delta}\n' \
               f'  equation={self.equation!r}\n' \
               f'  fit_method={self.fit_method!r}\n' \
               f'  max_iterations={self.max_iterations}\n' \
               f'  params={param_str}\n' \
               f'  second_derivs_B={self.second_derivs_B}\n' \
               f'  second_derivs_H={self.second_derivs_H}\n' \
               f'  tolerance={self.tolerance}\n' \
               f'  ux={np.array2string(self.ux, prefix="     ")}\n' \
               f'  uy={np.array2string(self.uy, prefix="     ")}\n' \
               f'  uy_weights_only={self.uy_weights_only}\n' \
               f'  weighted={self.weighted}\n' \
               f'  x={np.array2string(self.x, prefix="    ")}\n' \
               f'  y={np.array2string(self.y, prefix="    ")}\n' \
               f')'


@dataclass(eq=False, order=False, frozen=True)
class Result:
    """The result from a fit model."""

    calls: int
    """The number of calls to the DLL fit function."""

    chisq: float
    """The chi-squared value."""

    correlation: np.ndarray[float]
    """Parameter correlation coefficient matrix."""

    covariance: np.ndarray[float]
    """Parameter covariance matrix."""

    dof: int
    """The number of degrees of freedom."""

    eof: float
    """The error-of-fit value (the standard deviation of the residuals)."""

    iterations: int
    """The total number of fit iterations."""

    params: ResultParameters
    """The result parameters from the fit model."""

    def __repr__(self) -> str:
        # add indentation to the numpy arrays
        cor_str = np.array2string(self.correlation, prefix=' ' * 14)
        cov_str = np.array2string(self.covariance, prefix=' ' * 13)

        # add indentation to the parameters
        if not self.params:
            param_str = 'ResultParameters()'
        else:
            indent = ' ' * 4
            params = [indent]
            params.extend(str(self.params).splitlines())
            params[-1] = ')'
            param_str = f'\n{indent}'.join(params)

        return f'Result(\n' \
               f'  calls={self.calls}\n' \
               f'  chisq={self.chisq}\n' \
               f'  correlation={cor_str}\n' \
               f'  covariance={cov_str}\n' \
               f'  dof={self.dof}\n' \
               f'  eof={self.eof}\n' \
               f'  iterations={self.iterations}\n' \
               f'  params={param_str}\n' \
               f')'

    def to_ureal(self,
                 *,
                 with_future: bool = False,
                 label: str = 'future') -> list[UncertainReal]:
        r"""Convert the result to a correlated ensemble of
        :ref:`uncertain real numbers <uncertain_real_number>`.

        Parameters
        ----------
        with_future
            Whether to include an :ref:`uncertain real number <uncertain_real_number>`
            in the ensemble that is a *future* indication in response to a given
            stimulus (a predicted future response). This reflects the variability
            of single indications as well as the underlying uncertainty in the fit
            parameters. The *value* of this *future* uncertain number is zero,
            and the *uncertainty* component is :math:`\sqrt{\frac{\chi^2}{dof}}`.
        label
            The label to assign to the *future* uncertain number.

        Returns
        -------
        :class:`list` of :class:`~lib.UncertainReal`
            A correlated ensemble of
            :ref:`uncertain real numbers <uncertain_real_number>`.

        Examples
        --------
        Suppose the sample data has a linear relationship

            >>> x = [3, 7, 11, 15, 18, 27, 29, 30, 30, 31, 31, 32, 33, 33, 34, 36,
            ...      36, 36, 37, 38, 39, 39, 39, 40, 41, 42, 42, 43, 44, 45, 46, 47, 50]
            >>> y = [5, 11, 21, 16, 16, 28, 27, 25, 35, 30, 40, 32, 34, 32, 34, 37,
            ...      38, 34, 36, 38, 37, 36, 45, 39, 41, 40, 44, 37, 44, 46, 46, 49, 51]

        The intercept and slope are determined from a fit

            >>> from msl.nlf import LinearModel
            >>> with LinearModel() as model:
            ...    result = model.fit(x, y)

        We can estimate the response to a particular stimulus, say :math:`x=21.5`

            >>> intercept, slope = result.to_ureal()
            >>> intercept + 21.5*slope
            ureal(23.257962225044...,0.82160705888850...,31.0)

        or a single future indication in response to a given stimulus may also be of
        interest (again, at :math:`x=21.5`)

            >>> intercept, slope, future = result.to_ureal(with_future=True)
            >>> intercept + 21.5*slope + future
            ureal(23.257962225044...,3.33240925795711...,31.0)

        The value here is the same as above (because the stimulus is the same),
        but the uncertainty is much larger, reflecting the variability of single
        indications as well as the underlying uncertainty in the intercept and
        slope.
        """
        if multiple_ureal is None:
            raise OSError('GTC is not installed, run: pip install GTC')

        # create ensemble
        values = self.params.values()
        uncerts = self.params.uncerts()
        labels = self.params.labels()
        if with_future:
            values = np.append(values, 0.0)
            uncerts = np.append(uncerts, np.sqrt(self.chisq/float(self.dof)))
            labels.append(label)
        ensemble = multiple_ureal(values, uncerts, self.dof, label_seq=labels)

        # set correlations
        for i, row in enumerate(self.correlation):
            for j, value in enumerate(row[i+1:], start=i+1):
                set_correlation(value, ensemble[i], ensemble[j])

        return ensemble
