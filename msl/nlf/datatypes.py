"""
Various data classes and enums.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np

try:
    from GTC import multiple_ureal
    from GTC import set_correlation
    from GTC.lib import UncertainReal
except ImportError:
    multiple_ureal = None
    set_correlation = None
    UncertainReal = 'UncertainReal'

from .parameter import InputParameters
from .parameter import ResultParameters


class FitMethod(Enum):
    """Fitting methods."""
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

    def __repr__(self):
        # add indentation to the numpy array
        coeff_str = np.array2string(self.coefficients, prefix=' ' * 15)

        return f'Correlation(\n' \
               f'  coefficients={coeff_str}\n' \
               f'  path={self.path!r}\n' \
               f')'


@dataclass(eq=False, order=False, frozen=True)
class Correlations:
    """Information about the correlations for a fit model."""

    data: List[Correlation]
    """A :class:`list` of :class:`.Correlation` objects."""

    is_correlated: np.ndarray[bool]
    """Indicates which variables are correlated. The index 0 corresponds
    to the `y`-variable, the index 1 to `x1`, 2 to `x2`, etc."""

    def __repr__(self):
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

    fitting_method: str
    """The fitting method that is used. See :class:`.FitMethod`."""

    max_iterations: int
    """The maximum number of fit iterations allowed."""

    params: InputParameters
    """The input parameters to the fit model."""

    second_derivs_B: bool
    """Whether the second derivatives in the **B** matrix are used in the 
    fitting process."""

    second_derivs_H: bool
    """Whether the second derivatives in the **H** (Hessian) matrix are
    used in the fitting process."""

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
    """The independent variable(s) (stimulus)."""

    y: np.ndarray[float]
    """The dependent variable (response)."""

    def __repr__(self):
        indent = ' ' * 4

        # add indentation to the correlations
        corr = [indent]
        corr.extend(str(self.correlations).splitlines())
        corr[-1] = ')'
        corr_str = f'\n{indent}'.join(corr)

        # add indentation to the parameters
        params = [indent]
        params.extend(str(self.params).splitlines())
        params[-1] = ')'
        param_str = f'\n{indent}'.join(params)

        return f'Input(\n' \
               f'  correlated={self.correlated}\n' \
               f'  correlations={corr_str}\n' \
               f'  delta={self.delta}\n' \
               f'  equation={self.equation!r}\n' \
               f'  fitting_method={self.fitting_method!r}\n' \
               f'  max_iterations={self.max_iterations}\n' \
               f'  params={param_str}\n' \
               f'  second_derivs_B={self.second_derivs_B}\n' \
               f'  second_derivs_H={self.second_derivs_H}\n' \
               f'  tolerance={self.tolerance}\n' \
               f'  ux={np.array2string(self.ux, prefix="     ")}\n' \
               f'  uy={self.uy}\n' \
               f'  uy_weights_only={self.uy_weights_only}\n' \
               f'  weighted={self.weighted}\n' \
               f'  x={np.array2string(self.x, prefix="    ")}\n' \
               f'  y={self.y}\n' \
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

    def __repr__(self):
        # add indentation to the numpy arrays
        cor_str = np.array2string(self.correlation, prefix=' ' * 14)
        cov_str = np.array2string(self.covariance, prefix=' ' * 13)

        # add indentation to the parameters
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
                 label: str = 'future') -> List[UncertainReal]:
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
