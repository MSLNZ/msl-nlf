"""Various data classes and enums."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

try:
    from GTC import multiple_ureal, set_correlation  # type: ignore[import-untyped]
except ModuleNotFoundError:
    multiple_ureal = None
    set_correlation = None

if TYPE_CHECKING:
    from numpy.typing import NDArray

    try:
        from GTC.lib import UncertainReal  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        UncertainReal = object

    from .parameters import InputParameters, ResultParameters


PI = str(np.pi)


class FitMethod(Enum):
    """Fitting methods.

    Least squares (LS) minimises the sum of the squares of the vertical distances
    between each point and the fitted curve. The algorithms implemented
    for Levenberg Marquardt, Amoeba and Powell are described in
    [Numerical Recipes](http://numerical.recipes/){:target="_blank"}.

    Minimum distance (MD) minimises the sum of the distances (in two dimensions)
    between each point and the fitted curve. This type of fit is not available
    when data is correlated nor is it available when there is more than one
    independent variable (stimulus).

    MiniMax (MM) minimises the value of the maximum absolute y-residual. This
    type of fit is not available when data is correlated.

    Attributes:
        LM: Levenberg-Marquardt.
        AMOEBA_LS: Amoeba least squares.
        AMOEBA_MD: Amoeba minimum distance.
        AMOEBA_MM: Amoeba minimax.
        POWELL_LS: Powell least squares.
        POWELL_MD: Powell minimum distance.
        POWELL_MM: Powell minimax.
    """

    LM: str = "Levenberg-Marquardt"
    AMOEBA_LS: str = "Amoeba least squares"
    AMOEBA_MD: str = "Amoeba minimum distance"
    AMOEBA_MM: str = "Amoeba minimax"
    POWELL_LS: str = "Powell least squares"
    POWELL_MD: str = "Powell minimum distance"
    POWELL_MM: str = "Powell minimax"


class ResidualType(Enum):
    """Residual Type that is used to evaluate the error-of-fit value (the standard deviation of the residuals).

    Attributes:
        DX_X: Uncertainty in $x$ versus $x$.
        DX_Y: Uncertainty in $x$ versus $y$.
        DY_X: Uncertainty in $y$ versus $x$.
        DY_Y: Uncertainty in $y$ versus $y$.
    """

    DX_X: str = "dx v x"
    DX_Y: str = "dx v y"
    DY_X: str = "dy v x"
    DY_Y: str = "dy v y"


@dataclass(eq=False, order=False, frozen=True)
class Correlation:
    """Information about correlation coefficients.

    Attributes:
        path: The path to the correlation file.
        coefficients: The correlation coefficients.
    """

    path: str
    coefficients: NDArray[np.float64]

    def __repr__(self) -> str:
        """Return object representation."""
        # add indentation to the numpy array
        coeff_str = np.array2string(self.coefficients, prefix=" " * 15)

        return f"Correlation(\n  coefficients={coeff_str}\n  path={self.path!r}\n)"


@dataclass(eq=False, order=False, frozen=True)
class Correlations:
    """Information about the correlations for a fit model.

    Attributes:
        data: A [list][] of [Correlation][msl.nlf.datatypes.Correlation] objects.
        is_correlated: Indicates which variables are correlated. The index 0
            corresponds to the `y`-variable, the index 1 to `x1`, 2 to `x2`, etc.
    """

    data: list[Correlation]
    is_correlated: NDArray[np.bool_]

    def __repr__(self) -> str:
        """Return object representation."""
        # add indentation to the data
        data_str = ""
        if len(self.data) > 0:
            indent = " " * 4
            lines = [indent]
            for item in self.data:
                lines.extend(str(item).splitlines())
                lines[-1] += ","
            lines[-1] = ")"
            data_str = f"\n{indent}".join(lines)

        # add indentation to the numpy array
        corr_str = np.array2string(self.is_correlated, prefix=" " * 16)

        return f"Correlations(\n  data=[{data_str}]\n  is_correlated={corr_str}\n)"


@dataclass(eq=False, order=False, frozen=True)
class Input:
    """The input data to a fit model.

    Attributes:
        absolute_residuals: Whether absolute residuals or relative residuals are used
            to evaluate the error-of-fit value (the standard deviation of the residuals).
        correlated: Whether correlations are applied in the fitting process.
        correlations: The information about the correlation coefficients.
        delta: Only used for Amoeba fitting.
        equation: The equation of the fit model.
        fit_method: The method that is used for the fit.
        max_iterations: The maximum number of fit iterations allowed.
        params: The input parameters to the fit model.
        residual_type: Residual Type that is used to evaluate the error-of-fit value
            (the standard deviation of the residuals).
        second_derivs_B: Whether the second derivatives in the **B** matrix are included in
            the propagation of uncertainty calculations.
        second_derivs_H: Whether the second derivatives in the curvature matrix, **H** (Hessian),
            are included in the propagation of uncertainty calculations.
        tolerance: The tolerance value to stop the fitting process.
        ux: Standard uncertainties in the x (stimulus) data.
        uy: Standard uncertainties in the y (response) data.
        uy_weights_only: Whether the y uncertainties only or a combination of the x and y
            uncertainties are used to calculate the weights for a weighted fit.
        weighted: Whether to include the standard uncertainties in the fitting
            process to perform a weighted fit.
        x: The independent variable(s) (stimulus) data.
        y: The dependent variable (response) data.
    """

    absolute_residuals: bool
    correlated: bool
    correlations: Correlations
    delta: float
    equation: str
    fit_method: FitMethod
    max_iterations: int
    params: InputParameters
    residual_type: ResidualType
    second_derivs_B: bool  # noqa: N815
    second_derivs_H: bool  # noqa: N815
    tolerance: float
    ux: NDArray[np.float64]
    uy: NDArray[np.float64]
    uy_weights_only: bool
    weighted: bool
    x: NDArray[np.float64]
    y: NDArray[np.float64]

    def __repr__(self) -> str:
        """Return object representation."""
        indent = " " * 4

        # add indentation to the correlations
        corr = [indent]
        corr.extend(str(self.correlations).splitlines())
        corr[-1] = ")"
        corr_str = f"\n{indent}".join(corr)

        # add indentation to the parameters
        if not self.params:
            param_str = "InputParameters()"
        else:
            params = [indent]
            params.extend(str(self.params).splitlines())
            params[-1] = ")"
            param_str = f"\n{indent}".join(params)

        return (
            f"Input(\n"
            f"  absolute_residuals={self.absolute_residuals}\n"
            f"  correlated={self.correlated}\n"
            f"  correlations={corr_str}\n"
            f"  delta={self.delta}\n"
            f"  equation={self.equation!r}\n"
            f"  fit_method={self.fit_method!r}\n"
            f"  max_iterations={self.max_iterations}\n"
            f"  params={param_str}\n"
            f"  residual_type={self.residual_type!r}\n"
            f"  second_derivs_B={self.second_derivs_B}\n"
            f"  second_derivs_H={self.second_derivs_H}\n"
            f"  tolerance={self.tolerance}\n"
            f"  ux={np.array2string(self.ux, prefix='     ')}\n"
            f"  uy={np.array2string(self.uy, prefix='     ')}\n"
            f"  uy_weights_only={self.uy_weights_only}\n"
            f"  weighted={self.weighted}\n"
            f"  x={np.array2string(self.x, prefix='    ')}\n"
            f"  y={np.array2string(self.y, prefix='    ')}\n"
            f")"
        )


@dataclass(eq=False, order=False, frozen=False)
class Result:
    r"""The result from a fit model.

    Attributes:
        chisq: The chi-squared value.
        correlation: Parameter correlation coefficient matrix.
        covariance: Parameter covariance matrix.
        dof: The number of degrees of freedom that are retained. If a fit is weighted or correlated,
            the degrees of freedom is $+\infty$. Otherwise, the degrees of freedom is equal to the
            number of data points (observations) minus the number of fit parameters.
        eof: The error-of-fit value (the standard deviation of the residuals).
        iterations: The total number of fit iterations.
        num_calls: The number of times that the fit function in the shared library was recursively
            called with updated best-fit parameters.
        params: The result parameters from the fit model.
    """

    chisq: float
    correlation: NDArray[np.float64]
    covariance: NDArray[np.float64]
    dof: float
    eof: float
    iterations: int
    num_calls: int
    params: ResultParameters

    def __repr__(self) -> str:
        """Return object representation."""
        # add indentation to the numpy arrays
        cor_str = np.array2string(self.correlation, prefix=" " * 14)
        cov_str = np.array2string(self.covariance, prefix=" " * 13)

        # add indentation to the parameters
        if not self.params:
            param_str = "ResultParameters()"
        else:
            indent = " " * 4
            params = [indent]
            params.extend(str(self.params).splitlines())
            params[-1] = ")"
            param_str = f"\n{indent}".join(params)

        return (
            f"Result(\n"
            f"  chisq={self.chisq}\n"
            f"  correlation={cor_str}\n"
            f"  covariance={cov_str}\n"
            f"  dof={self.dof}\n"
            f"  eof={self.eof}\n"
            f"  iterations={self.iterations}\n"
            f"  num_calls={self.num_calls}\n"
            f"  params={param_str}\n"
            f")"
        )

    def to_ureal(self, *, with_future: bool = False, label: str = "future") -> list[UncertainReal]:
        r"""Convert the result to a correlated ensemble of [uncertain real numbers][uncertain_real_number]{:target="_blank"}.

        Args:
            with_future:
                Whether to include an [uncertain real numbers][uncertain_real_number]{:target="_blank"}
                in the ensemble that is a *future* indication in response to a given
                stimulus (a predicted future response). This reflects the variability
                of single indications as well as the underlying uncertainty in the fit
                parameters. The *value* of this *future* uncertain number is zero,
                and the *uncertainty* component is $\sqrt{\frac{\chi^2}{dof}}$.
            label:
                The label to assign to the *future* uncertain number.

        Returns:
            A correlated ensemble of [uncertain real numbers][uncertain_real_number]{:target="_blank"}

        Examples:
            Suppose the sample data has a linear relationship

            >>> x = [3, 7, 11, 15, 18, 27, 29, 30, 30, 31, 31, 32, 33, 33, 34, 36,
            ...      36, 36, 37, 38, 39, 39, 39, 40, 41, 42, 42, 43, 44, 45, 46, 47, 50]
            >>> y = [5, 11, 21, 16, 16, 28, 27, 25, 35, 30, 40, 32, 34, 32, 34, 37,
            ...      38, 34, 36, 38, 37, 36, 45, 39, 41, 40, 44, 37, 44, 46, 46, 49, 51]

            The intercept and slope are determined from a fit

            >>> from msl.nlf import LinearModel
            >>> with LinearModel() as model:
            ...    result = model.fit(x, y)

            <!-- skip: start if(no_gtc, reason="GTC cannot be imported") -->

            We can estimate the response to a particular stimulus, say $x=21.5$

            >>> intercept, slope = result.to_ureal()
            >>> intercept + 21.5*slope
            ureal(23.257962225044...,0.82160705888850...,31.0)

            or a single future indication in response to a given stimulus may also be of interest (again, at $x=21.5$)

            >>> intercept, slope, future = result.to_ureal(with_future=True)
            >>> intercept + 21.5*slope + future
            ureal(23.257962225044...,3.33240925795711...,31.0)

            The value here is the same as above (because the stimulus is the same),
            but the uncertainty is much larger, reflecting the variability of single
            indications as well as the underlying uncertainty in the intercept and
            slope.
        """  # noqa: E501
        if multiple_ureal is None:
            msg = "GTC is not installed, run: pip install GTC"
            raise OSError(msg)

        # create ensemble
        values = self.params.values()
        uncerts = self.params.uncerts()
        labels = self.params.labels()
        if with_future:
            values = np.append(values, 0.0)
            uncerts = np.append(uncerts, np.sqrt(self.chisq / float(self.dof)))
            labels.append(label)

        ensemble: list[UncertainReal] = multiple_ureal(values, uncerts, self.dof, label_seq=labels)

        # set correlations
        for i, row in enumerate(self.correlation):
            for j, value in enumerate(row[i + 1 :], start=i + 1):
                # cast the value to float so that it is not of type np.float
                set_correlation(float(value), ensemble[i], ensemble[j])

        return ensemble
