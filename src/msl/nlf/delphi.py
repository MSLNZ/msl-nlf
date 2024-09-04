"""Wrapper around the Delphi functions."""

from __future__ import annotations

import os
import sys
import sysconfig
from ctypes import CDLL, POINTER, byref, c_bool, c_double, c_int, c_wchar_p, create_string_buffer, create_unicode_buffer
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from .types import (
        ArrayLike1D,
        ArrayLike2D,
        CtypesOrNumpyBool,
        CtypesOrNumpyDouble,
        EvaluateArray,
        GetFunctionValue,
        PBoolParamData,
        PData,
        PIsCorrelated,
        PMultiData,
        PRealParamData,
        PSquareMatrix,
        UserDefinedDict,
    )


NPTS = 100001  # max points
NPAR = 99  # max parameters
NVAR = 30  # max variables

# It is easier (and faster) to use a 1D array rather than a 2D array when
# passing arrays between the 64-bit client and the 32-bit server. This is
# valid since the arrays are C_CONTIGUOUS.
multi_data = c_double * (NPTS * NVAR)
data = c_double * NPTS
real_param_data = c_double * NPAR
bool_param_data = c_bool * NPAR
is_correlated = c_bool * ((NVAR + 1) * (NVAR + 1))

# Except for the covariance matrix, keep it 2D since it does not get sent
# by the 64-bit client, instead it gets created in the 32-bit server and
# returned to the 64-bit client as a list[list[float]]
square_matrix = (c_double * NPAR) * NPAR

here = Path(__file__).parent
filename_map: dict[str, Path] = {
    "win32": here / "bin" / "nlf-windows-i386.dll",
    "win-amd64": here / "bin" / "nlf-windows-x86_64.dll",
}


def fit(  # noqa: PLR0913
    *,
    lib: CDLL,
    cfg_path: str,
    equation: bytes,
    weighted: bool,
    x: CtypesOrNumpyDouble,
    y: CtypesOrNumpyDouble,
    ux: CtypesOrNumpyDouble,
    uy: CtypesOrNumpyDouble,
    npts: int,
    nvars: int,
    a: CtypesOrNumpyDouble,
    constant: CtypesOrNumpyBool,
    covar: ArrayLike2D,
    ua: CtypesOrNumpyDouble,
    correlated: bool,
    is_corr_array: CtypesOrNumpyBool,
    corr_dir: str,
    max_iterations: int,
    nparams: int,
) -> dict[str, Any]:
    """Call the *DoNonlinearFit* function.

    Args:
        lib: The library instance.
        cfg_path: Path to the configuration file.
        equation: The fit equation.
        weighted: Whether to do a weighted fit.
        x: The independent variable (stimulus) data.
        y: The dependent variable (response) data.
        ux: The x uncertainties.
        uy: The y uncertainties.
        npts: The number of points.
        nvars: The number of variables.
        a: The fit parameters.
        constant: Whether each parameter if held fixed during the fit.
        covar: Array to write the covariance matrix.
        ua: Array to write the uncertainty of each parameter.
        correlated: Whether to perform a correlated fit.
        is_corr_array: Which parameters are correlated.
        corr_dir: The directory that contains the correlation matrices.
        max_iterations: The maximum number of fit iterations.
        nparams: The number of fit parameters.

    Returns:
        The fit results.
    """
    calls = 0
    iter_total = 0
    iterations = c_int()
    chisq = c_double()
    eof = c_double()
    error = c_bool()
    error_str = create_unicode_buffer(1024)

    # the ResetFile function was added in v5.43
    lib.ResetFile()

    while iter_total < max_iterations:
        calls += 1
        lib.DoNonlinearFit(
            cfg_path,
            equation,
            weighted,
            x,
            y,
            ux,
            uy,
            npts,
            nvars,
            a,
            constant,
            covar,
            ua,
            correlated,
            is_corr_array,
            corr_dir,
            chisq,
            eof,
            iterations,
            error,
            error_str,
        )

        if error.value:
            raise RuntimeError(error_str.value)

        iter_total += iterations.value
        if iterations.value <= 3:  # noqa: PLR2004
            # According to the "Nonlinear Fitting Software Instructions" manual:
            #   Once the iterations have stopped, it is good practice to click
            #   again on the Calculate button using the newly found parameters
            #   as starting values until number of iterations stops at 2 or 3
            break

    if hasattr(covar, "dtype"):
        covar = covar[:nparams, :nparams]  # type: ignore[call-overload]
    else:
        covar = [[covar[i][j] for i in range(nparams)] for j in range(nparams)]

    return {
        "a": a[:nparams],
        "ua": ua[:nparams],
        "covariance": covar,
        "chisq": chisq.value,
        "eof": eof.value,
        "iterations": iter_total,
        "num_calls": calls,
    }


def delphi_version(lib: CDLL) -> str:
    """Call the *GetVersion* function.

    Args:
        lib: The library instance.

    Returns:
        The version number of the shared library.
    """
    # the GetVersion function was added in v5.41
    buffer = create_unicode_buffer(16)
    lib.GetVersion.restype = None
    lib.GetVersion.argtypes = [c_wchar_p]
    lib.GetVersion(buffer)
    return buffer.value


def define_fit_fcn(lib: CDLL, *, as_ctypes: bool) -> None:
    """Defines the *argtypes* and *restype* of the *DoNonlinearFit* function.

    Args:
        lib: The library instance.
        as_ctypes: Whether [ctypes][] arrays or [numpy.ndarray][]s will be passed to the `DoNonlinearFit` function.
    """
    p_multi_data: PMultiData
    p_data: PData
    p_real_param_data: PRealParamData
    p_bool_param_data: PBoolParamData
    p_square_matrix: PSquareMatrix
    p_is_correlated: PIsCorrelated

    if as_ctypes:
        p_multi_data = POINTER(multi_data)
        p_data = POINTER(data)
        p_real_param_data = POINTER(real_param_data)
        p_bool_param_data = POINTER(bool_param_data)
        p_square_matrix = POINTER(square_matrix)
        p_is_correlated = POINTER(is_correlated)
    else:
        from numpy.ctypeslib import ndpointer as p

        p_multi_data = p(dtype=c_double, shape=(NVAR, NPTS), flags="C_CONTIGUOUS")
        p_data = p(dtype=c_double, shape=(NPTS,), flags="C_CONTIGUOUS")
        p_real_param_data = p(dtype=c_double, shape=(NPAR,), flags="C_CONTIGUOUS")
        p_bool_param_data = p(dtype=c_bool, shape=(NPAR,), flags="C_CONTIGUOUS")
        p_square_matrix = p(dtype=c_double, shape=(NPAR, NPAR), flags="C_CONTIGUOUS")
        p_is_correlated = p(dtype=c_bool, shape=(NVAR + 1, NVAR + 1), flags="C_CONTIGUOUS")

    lib.DoNonlinearFit.restype = None
    lib.DoNonlinearFit.argtypes = [
        c_wchar_p,  # ConfigFile:PChar
        c_char_p,  # EquationStr:PChar
        c_bool,  # WeightedFit:Boolean (added in v5.44)
        p_multi_data,  # xData:PMultiData
        p_data,  # yData:PData
        p_multi_data,  # xUn:PMultiData
        p_data,  # yUn:PData
        c_int,  # N:Integer
        c_int,  # xVar:Integer
        p_real_param_data,  # Params:PRealParamData
        p_bool_param_data,  # Constant:PBoolParamData
        p_square_matrix,  # Cov:PSquareMatrix
        p_real_param_data,  # ParamUncerts:PRealParamData
        c_bool,  # CorrData:Boolean
        p_is_correlated,  # IsCorrelated:PIsCorrelated
        c_wchar_p,  # CorrCoeffsDirectory:PChar
        POINTER(c_double),  # var ChiSquared:Double
        POINTER(c_double),  # var EofFit:Double
        POINTER(c_int),  # var NumIterations:Integer
        POINTER(c_bool),  # var Error:Boolean
        c_wchar_p,  # ErrorStr:PChar
    ]


@dataclass
class UserDefined:
    """A user-defined function.

    Attributes:
        equation: The value to use as the *equation* for a [Model][msl.nlf.model.Model].
        function: A reference to the *GetFunctionValue* function.
        name: The name returned by the *GetFunctionName* function.
        num_parameters: The value returned by the *GetNumParameters* function.
        num_variables: The value returned by the *GetNumVariables* function.
    """

    equation: str
    function: GetFunctionValue
    name: str
    num_parameters: int
    num_variables: int

    def to_dict(self) -> UserDefinedDict:
        """Convert this object to be a pickleable [dict][].

        The value of `function` is always [None][].
        """
        return {
            "equation": self.equation,
            "function": None,
            "name": self.name,
            "num_parameters": self.num_parameters,
            "num_variables": self.num_variables,
        }


def get_user_defined(directory: str | Path, extension: str) -> dict[Path, UserDefined]:
    """Get all user-defined functions.

    Args:
        directory: The directory to look for the user-defined functions.
        extension: The file extension for the user-defined functions.

    Returns:
        The user-defined functions.
    """
    n = c_int()
    buffer = create_string_buffer(255)

    functions: dict[Path, UserDefined] = {}
    for file in Path(directory).glob(f"*{extension}"):
        try:
            lib = CDLL(str(file))
        except OSError:
            # perhaps loading a library of the wrong bitness
            continue

        try:
            lib.GetFunctionName(buffer)
        except AttributeError:
            continue

        try:
            lib.GetNumParameters(byref(n))
        except AttributeError:
            continue
        else:
            num_par = n.value

        try:
            lib.GetNumVariables(byref(n))
        except AttributeError:
            continue
        else:
            num_var = n.value

        try:
            _ = lib.GetFunctionValue
        except AttributeError:
            continue

        name = buffer.value.decode(encoding="ansi", errors="replace")
        equation, *rest = name.split(":")

        functions[file] = UserDefined(
            equation=equation, function=lib.GetFunctionValue, name=name, num_parameters=num_par, num_variables=num_var
        )

    return functions


def evaluate(
    fcn: GetFunctionValue,
    a: ArrayLike1D,
    x: ArrayLike1D,
    shape: tuple[int, int],
    y: EvaluateArray,
) -> EvaluateArray:
    """Call *GetFunctionValue* in the user-defined function.

    Args:
        fcn: Reference to *GetFunctionValue*.
        a: Parameter values.
        x: Independent variable (stimulus) data. The data must already be transposed and flat.
        shape: The shape of the `x` data.
        y: Pre-allocated array for the dependent variable (response) data.

    Returns:
        Dependent variable (response) data.
    """
    j = 0
    y_val = c_double()
    nvars, npts = shape
    if hasattr(a, "dtype"):
        # a, x and y are np.ndarray
        for i in range(npts):
            fcn(x[j : j + nvars], a, y_val)
            y[i] = y_val.value
            j += nvars
    else:
        a_ref = (len(a) * c_double)(*a)
        x_ref = nvars * c_double
        for i in range(npts):
            fcn(x_ref(*x[j : j + nvars]), a_ref, y_val)
            y[i] = y_val.value
            j += nvars
    return y


def nlf_info(*, win32: bool = False) -> tuple[Path, bool, str]:
    """Get information about the non-linear fitting library.

    Args:
        win32: Whether to load the 32-bit Windows library.

    Returns:
        The path to the non-linear fitting library, whether to load the library using
        inter-process communication and the file extension for a user-defined function.
    """
    if win32 and sys.platform != "win32":
        msg = "Enabling the 'win32' feature is only supported on Windows"
        raise ValueError(msg)

    if sys.platform == "win32":
        if win32:
            return filename_map["win32"], sys.maxsize > 2**32, ".dll"
        return filename_map[sysconfig.get_platform()], False, ".dll"

    # must be Unix
    uname = os.uname()

    msg = (
        f"The non-linear-fitting library is not available for:\n"
        f"sysname={uname.sysname}\n"
        f"release={uname.release}\n"
        f"version={uname.version}\n"
        f"machine={uname.machine}\n"
    )
    raise ValueError(msg)
