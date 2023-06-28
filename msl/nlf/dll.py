"""
Wrapper around DLL functions.
"""
from __future__ import annotations

import os
from array import array
from ctypes import CDLL
from ctypes import POINTER
from ctypes import byref
from ctypes import c_bool
from ctypes import c_double
from ctypes import c_int
from ctypes import c_wchar_p
from ctypes import create_string_buffer
from ctypes import create_unicode_buffer
from dataclasses import dataclass
from typing import Callable
from typing import Sequence

__all__ = ('NPTS', 'NPAR', 'NVAR',
           'fit', 'version', 'define_fit_fcn',
           'real_param_data', 'square_matrix',
           'UserDefined', 'get_user_defined', 'evaluate')

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


def fit(dll: CDLL, **k) -> dict:
    """Call the *DoNonlinearFit* function in the DLL.

    Parameters
    ----------
    dll
        The instance of the DLL.
    k
        Keyword arguments that are required to perform the fit.

    Returns
    -------
    dict
        The fit results of the DLL.
    """
    calls = 0
    iter_total = 0
    max_iter = k['max_iterations']
    iterations = c_int()
    chisq = c_double()
    eof = c_double()
    error = c_bool()
    error_str = create_unicode_buffer(1024)
    while iter_total < max_iter:
        calls += 1
        dll.DoNonlinearFit(
            k['cfg_path'], k['equation'], k['x'], k['y'], k['ux'], k['uy'],
            k['npts'], k['nvars'], k['a'], k['constant'], k['covar'], k['ua'],
            k['correlated'], k['is_corr_array'], k['corr_dir'], chisq, eof,
            iterations, error, error_str)

        if error.value:
            raise RuntimeError(error_str.value)

        iter_total += iterations.value
        if iterations.value <= 3:
            # According to the "Nonlinear Fitting Software Instructions" manual:
            #   Once the iterations have stopped, it is good practice to click
            #   again on the Calculate button using the newly found parameters
            #   as starting values until number of iterations stops at 2 or 3
            break

    n, c = k['nparams'], k['covar']
    if hasattr(c, 'dtype'):
        covar = c[:n, :n]

        # In the Delphi code, the software doesn't attempt to calculate a value
        # for ua[i] and it puts a blank cell into the Results spreadsheet. But
        # if a fit had previously been carried out with a[i] varying, then ua[i]
        # will have the previous value because that part of the covariance matrix
        # doesn't get overwritten.
        k['ua'][k['constant']] = 0.0
        ua = k['ua'][:n]
    else:
        covar = [[c[i][j] for i in range(n)] for j in range(n)]
        constant = k['constant'].contents
        ua = [0.0 if constant[i] else u for i, u in enumerate(k['ua'][:n])]

    return {
        'a': k['a'][:n],
        'ua': ua,
        'covariance': covar,
        'chisq': chisq.value,
        'eof': eof.value,
        'iterations': iter_total,
        'calls': calls,
    }


def version(dll: CDLL) -> str:
    """Call the *GetVersion* function in the DLL.

    The *GetVersion* function was added to the DLL in v5.41. If an older
    version of the DLL is loaded when a :class:`~msl.nlf.model.Model` is
    created, calling this method will raise an :exc:`AttributeError`.

    Parameters
    ----------
    dll
        The instance of the DLL.

    Returns
    -------
    str
        The version number of the DLL.
    """
    buffer = create_unicode_buffer(16)
    dll.GetVersion.restype = None
    dll.GetVersion.argtypes = [c_wchar_p]
    dll.GetVersion(buffer)
    return buffer.value


def define_fit_fcn(dll: CDLL, as_ctypes: bool) -> None:
    """Defines the *argtypes* and *restype* of the *DoNonlinearFit* function.

    Parameters
    ----------
    dll
        The instance of the DLL.
    as_ctypes
        Whether :mod:`ctypes` arrays or :class:`numpy.ndarray`\\s will be
        passed to the `DoNonlinearFit` function.
    """
    if as_ctypes:
        p_multi_data = POINTER(multi_data)
        p_data = POINTER(data)
        p_real_param_data = POINTER(real_param_data)
        p_bool_param_data = POINTER(bool_param_data)
        p_square_matrix = POINTER(square_matrix)
        p_is_correlated = POINTER(is_correlated)
    else:
        import numpy.ctypeslib
        flag = 'C_CONTIGUOUS'
        p = numpy.ctypeslib.ndpointer
        p_multi_data = p(dtype=c_double, shape=(NVAR, NPTS), flags=flag)
        p_data = p(dtype=c_double, shape=(NPTS,), flags=flag)
        p_real_param_data = p(dtype=c_double, shape=(NPAR,), flags=flag)
        p_bool_param_data = p(dtype=c_bool, shape=(NPAR,), flags=flag)
        p_square_matrix = p(dtype=c_double, shape=(NPAR, NPAR), flags=flag)
        p_is_correlated = p(dtype=c_bool, shape=(NVAR+1, NVAR+1), flags=flag)

    dll.DoNonlinearFit.restype = None
    dll.DoNonlinearFit.argtypes = [
        c_wchar_p,          # ConfigFile:PChar
        c_wchar_p,          # EquationStr:PChar
        p_multi_data,       # xData:PMultiData
        p_data,             # yData:PData
        p_multi_data,       # xUn:PMultiData
        p_data,             # yUn:PData
        c_int,              # N:Integer
        c_int,              # xVar:Integer
        p_real_param_data,  # Params:PRealParamData
        p_bool_param_data,  # Constant:PBoolParamData
        p_square_matrix,    # Cov:PSquareMatrix
        p_real_param_data,  # ParamUncerts:PRealParamData
        c_bool,             # CorrData:Boolean
        p_is_correlated,    # IsCorrelated:PIsCorrelated
        c_wchar_p,          # CorrCoeffsDirectory:PChar
        POINTER(c_double),  # var ChiSquared:Double
        POINTER(c_double),  # var EofFit:Double
        POINTER(c_int),     # var NumIterations:Integer
        POINTER(c_bool),    # var Error:Boolean
        c_wchar_p           # ErrorStr:PChar
    ]


@dataclass
class UserDefined:
    """A user-defined function that has been compiled to a DLL."""

    equation: str
    """The value to use as the *equation* for a :class:`~msl.nlf.model.Model`."""

    function: Callable
    """A reference to the *GetFunctionValue* function."""

    name: str
    """The name returned by the *GetFunctionName* function."""

    num_parameters: int
    """The value returned by the *GetNumParameters* function."""

    num_variables: int
    """The value returned by the *GetNumVariables* function."""

    def to_dict(self) -> dict:
        """Convert this object to be a pickleable :class:`dict`.

        The value of :attr:`.function` becomes :data:`None`.
        """
        return {
            'equation': self.equation,
            'function': None,
            'name': self.name,
            'num_parameters': self.num_parameters,
            'num_variables': self.num_variables,
        }


def get_user_defined(directory: str) -> dict[str, UserDefined]:
    """Get all user-defined functions.

    Parameters
    ----------
    directory
        The directory to look for the user-defined functions.

    Returns
    -------
    :class:`dict` [ :class:`str`, :class:`.UserDefined` ]
        The keys are the filenames and the values are :class:`.UserDefined`.
    """
    functions = {}
    n = c_int()
    buffer = create_string_buffer(255)
    for filename in os.listdir(directory):
        _, ext = os.path.splitext(filename)
        if ext.lower() != '.dll':
            continue

        try:
            dll = CDLL(os.path.join(directory, filename))
        except OSError:
            # perhaps loading a DLL of the wrong bitness
            continue

        try:
            dll.GetFunctionName(buffer)
        except AttributeError:
            continue

        try:
            dll.GetNumParameters(byref(n))
        except AttributeError:
            continue
        else:
            num_par = n.value

        try:
            dll.GetNumVariables(byref(n))
        except AttributeError:
            continue
        else:
            num_var = n.value

        try:
            dll.GetFunctionValue
        except AttributeError:
            continue

        name = buffer.value.decode(encoding='ansi', errors='replace')
        equation, *rest = name.split(':')

        functions[filename] = UserDefined(
            equation=equation,
            function=dll.GetFunctionValue,
            name=name,
            num_parameters=num_par,
            num_variables=num_var
        )

    return functions


def evaluate(fcn: Callable,
             a: Sequence[float],
             x: Sequence[float],
             shape: tuple[int, int],
             y):
    """Call *GetFunctionValue* in the user-defined DLL.

    Parameters
    ----------
    fcn
        Reference to *GetFunctionValue*.
    a
        Parameter values.
    x
        Independent variable (stimulus) data. The data must already be
        transposed and flat.
    shape
        The shape of the *x* data.
    y
        Pre-allocated sequence for the dependent variable (response) data.

    Returns
    -------
    :class:`~array.array` or :class:`~numpy.ndarray`
        Dependent variable (response) data.
    """
    j = 0
    y_val = c_double()
    nvars, npts = shape
    if isinstance(a, array):
        a_ref = (len(a) * c_double)(*a)
        x_ref = nvars * c_double
        for i in range(npts):
            fcn(x_ref(*x[j:j+nvars]), a_ref, y_val)
            y[i] = y_val.value
            j += nvars
    else:
        # a, x and y are np.ndarray
        for i in range(npts):
            fcn(x[j:j+nvars], a, y_val)
            y[i] = y_val.value
            j += nvars
    return y
