"""
A model to use for a non-linear fit.
"""
from __future__ import annotations

import os
import re
import warnings
from shutil import rmtree
from tempfile import mkdtemp
from typing import Dict
from typing import Iterable
from typing import Literal
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import overload

import numpy as np
from msl.loadlib import IS_PYTHON_64BIT
from msl.loadlib import LoadLibrary

from .client_server import ClientNLF
from .datatypes import Correlation
from .datatypes import Correlations
from .datatypes import FitMethod
from .datatypes import Input
from .datatypes import Result
from .datatypes import ResultParameters
from .dll import NPAR
from .dll import NPTS
from .dll import NVAR
from .dll import define_fit_fcn
from .dll import fit
from .dll import version
from .parameter import InputParameterType
from .parameter import InputParameters

_n_params_regex = re.compile(r'a\d+')
_n_vars_regex = re.compile(r'(?<!e)(x\d*)')
_corr_var_regex = re.compile(r'^(Y|X\d{1,2})$')
_corr_file_regex = re.compile(r'^CorrCoeffs (Y|X\d{1,2})-(Y|X\d{1,2})\.txt$')

# array-like types
ArrayLike1D = Sequence[float]
"""A 1D array."""

ArrayLike = Union[ArrayLike1D, Sequence[Sequence[float]]]
"""A 1D or 2D array."""


def _fill_array(a, b):
    """Fill array *a* with the values in array *b*. Returns *b*."""
    if b is None:
        return

    b = np.asarray(b)
    if b.ndim == 1:
        if a.ndim == 1:
            a[:b.size] = b
        else:
            a[0, :b.size] = b
    elif b.ndim == 2:
        i, j = b.shape
        a[:i, :j] = b
    else:
        raise ValueError(f'An array of shape {b.shape} is not supported')

    return b


class Model:

    _np_map = {'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
               'exp': np.exp, 'ln': np.log, 'log': np.log10,
               'arcsin': np.arcsin, 'arcos': np.arccos}

    MAX_POINTS: int = NPTS
    """Maximum number of data points allowed."""

    MAX_PARAMETERS: int = NPAR
    """Maximum number of fit parameters allowed."""

    MAX_VARIABLES: int = NVAR
    """Maximum number of x (stimulus) variables allowed."""

    FitMethod = FitMethod
    """Fitting methods."""

    def __init__(self,
                 equation: str,
                 *,
                 dll: str = None,
                 **options) -> None:
        """A model for the non-linear fitting software of P. Saunders, MSL.

        Parameters
        ----------
        equation
            The fit equation. The x variables (stimulus) must be specified as
            *x1*, *x2*, etc. and the parameters as *a1*, *a2*, etc. If only
            one x-variable is required, it can be simply entered as x. The
            arithmetic operations and functions that are recognised are:

            .. centered::
                + – * / ^ sin cos tan exp ln log arcsin arcos

            where **^** indicates raising to the power. All white space is
            ignored in the equation. For example, to fit a general quadratic
            equation one would use ``a1+a2*x+a3*x^2``. The **sqrt** function
            can be written as **^0.5**, for example, **sqrt(2*x)** would be
            expressed as **(2*x)^0.5** in the equation.
        dll
            The path to a non-linear fit DLL file. A default DLL is chosen
            based on the bitness of the Python interpreter. If you want to
            load a 32-bit DLL in 64-bit Python then set *dll* to be **nlf32**.
            See :ref:`nlf_32vs64` for reasons why you may want to use a
            different DLL bitness. You may also specify a path to a DLL that
            is located in a particular directory of your computer.
        **options
            All additional keyword arguments are passed to :meth:`.options`.
        """
        self._dll = None
        self._tmp_dir = mkdtemp(prefix='nlf-')
        self._cfg_path = os.path.join(self._tmp_dir, 'options.cfg')
        self._equation = equation
        self._version = ''
        self._corr_dir = ''
        self._corr_dict = {}
        self._weighted = False
        self._correlated = False
        self._max_iterations = 0
        self._dll_path = dll
        self._show_warnings = True

        # these are re-defined in Model subclasses and are
        # used when creating a CompositeModel
        self._factor = ''
        self._offset = ''
        self._composite_equation = ''

        variables = set(_n_vars_regex.findall(equation))
        if 'x' in variables and 'x1' in variables:
            raise ValueError("Cannot use both 'x' and 'x1' in equation")
        if 'x0' in variables:
            raise ValueError("Cannot use 'x0' in equation")

        self._num_vars = len(variables)
        if self._num_vars > self.MAX_VARIABLES:
            raise ValueError(f'Too many x variables in equation '
                             f'[{self._num_vars} > {self.MAX_VARIABLES}]')

        params = set(_n_params_regex.findall(equation))
        if 'a0' in params:
            raise ValueError("Cannot use 'a0' in equation")

        self._num_params = len(params)
        if self._num_params > self.MAX_PARAMETERS:
            raise ValueError(f'Too many fitting parameters in equation '
                             f'[{self._num_params} > {self.MAX_PARAMETERS}]')

        self.options(**options)

        self._x = np.zeros((self.MAX_VARIABLES, self.MAX_POINTS))
        self._ux = np.zeros((self.MAX_VARIABLES, self.MAX_POINTS))
        self._y = np.zeros(self.MAX_POINTS)
        self._uy = np.zeros(self.MAX_POINTS)
        self._a = np.zeros(self.MAX_PARAMETERS)
        self._constant = np.zeros(self.MAX_PARAMETERS, dtype=bool)
        self._is_corr_array = np.zeros((self.MAX_VARIABLES + 1, self.MAX_VARIABLES + 1), dtype=bool)

        here = os.path.dirname(__file__)
        if not dll:
            bit = '64' if IS_PYTHON_64BIT else '32'
            dll = os.path.join(here, f'nlf{bit}.dll')
        elif dll == 'nlf32':
            dll = os.path.join(here, 'nlf32.dll')
        elif dll == 'nlf64':
            if not IS_PYTHON_64BIT:
                raise ValueError('Cannot load a 64-bit DLL in 32-bit Python')
            dll = os.path.join(here, 'nlf64.dll')

        self._dll_path = dll

        try:
            self._dll = LoadLibrary(dll, libtype='cdll')
        except OSError as e:
            if e.winerror == 193:
                # Tried to load a 32-bit DLL in 64-bit Python.
                # Use interprocess communication (see msl-loadlib).
                self._dll = ClientNLF(dll)
            else:
                raise
        else:
            self._covar = np.zeros((self.MAX_PARAMETERS, self.MAX_PARAMETERS))
            self._ua = np.zeros(self.MAX_PARAMETERS)
            define_fit_fcn(self._dll.lib, False)

    def __add__(self, rhs):
        op = '+'
        if isinstance(rhs, Model):
            return CompositeModel(op, self, rhs, dll=self._dll_path)
        elif isinstance(rhs, (float, int)):
            return Model(f'{self._equation}{op}{rhs}', dll=self._dll_path)
        raise TypeError(f'unsupported operand type(s) for {op}: '
                        f'{type(self).__name__!r} and {type(rhs).__name__!r}')

    def __radd__(self, lhs):
        op = '+'
        if isinstance(lhs, (float, int)):
            return Model(f'{lhs}{op}{self._equation}', dll=self._dll_path)
        raise TypeError(f'unsupported operand type(s) for {op}: '
                        f'{type(lhs).__name__!r} and {type(self).__name__!r}')

    def __sub__(self, rhs):
        op = '-'
        if isinstance(rhs, Model):
            return CompositeModel(op, self, rhs, dll=self._dll_path)
        elif isinstance(rhs, (float, int)):
            return Model(f'{self._equation}{op}{rhs}', dll=self._dll_path)
        raise TypeError(f'unsupported operand type(s) for {op}: '
                        f'{type(self).__name__!r} and {type(rhs).__name__!r}')

    def __rsub__(self, lhs):
        op = '-'
        if isinstance(lhs, (float, int)):
            return Model(f'{lhs}{op}{self._equation}', dll=self._dll_path)
        raise TypeError(f'unsupported operand type(s) for {op}: '
                        f'{type(lhs).__name__!r} and {type(self).__name__!r}')

    def __mul__(self, rhs):
        op = '*'
        if isinstance(rhs, Model):
            return CompositeModel(op, self, rhs, dll=self._dll_path)
        elif isinstance(rhs, (float, int)):
            return Model(f'({self._equation}){op}{rhs}', dll=self._dll_path)
        raise TypeError(f'unsupported operand type(s) for {op}: '
                        f'{type(self).__name__!r} and {type(rhs).__name__!r}')

    def __rmul__(self, lhs):
        op = '*'
        if isinstance(lhs, (float, int)):
            return Model(f'{lhs}{op}({self._equation})', dll=self._dll_path)
        raise TypeError(f'unsupported operand type(s) for {op}: '
                        f'{type(lhs).__name__!r} and {type(self).__name__!r}')

    def __truediv__(self, rhs):
        op = '/'
        if isinstance(rhs, Model):
            return CompositeModel(op, self, rhs, dll=self._dll_path)
        elif isinstance(rhs, (float, int)):
            return Model(f'({self._equation}){op}{rhs}', dll=self._dll_path)
        raise TypeError(f'unsupported operand type(s) for {op}: '
                        f'{type(self).__name__!r} and {type(rhs).__name__!r}')

    def __rtruediv__(self, lhs):
        op = '/'
        if isinstance(lhs, (float, int)):
            return Model(f'{lhs}{op}({self._equation})', dll=self._dll_path)
        raise TypeError(f'unsupported operand type(s) for {op}: '
                        f'{type(lhs).__name__!r} and {type(self).__name__!r}')

    def __del__(self):
        try:
            self._cleanup()
        except AttributeError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    def __repr__(self):
        dirname, basename = os.path.split(self._dll_path)
        if os.path.dirname(__file__) == dirname:
            path = basename
        else:
            path = self._dll_path
        return f'{self.__class__.__name__}' \
               f'(equation={self._equation!r}, dll={path!r})'

    def _cleanup(self) -> None:
        if isinstance(self._dll, ClientNLF):
            self._dll.shutdown_server32()
        rmtree(self._tmp_dir, ignore_errors=True)

    def _load_correlations(self, num_x_vars: int) -> Correlations:
        # load the correlation files
        num_corr = num_x_vars + 1  # include y-variable
        coeffs = []
        if self._corr_dir:
            for filename in os.listdir(self._corr_dir):
                match = _corr_file_regex.match(filename)
                if match:
                    path = os.path.join(self._corr_dir, filename)
                    coeffs.append(Correlation(path=path, coefficients=np.loadtxt(path)))

        return Correlations(
            is_correlated=self._is_corr_array[:num_corr, :num_corr],
            data=coeffs
        )

    def _load_options(self) -> dict:
        # load the options.cfg file
        options = {}
        with open(self._cfg_path) as f:
            for line in f:
                k, v = line.split('=')
                if k in ('absolute_res', 'residual_type', 'show_info_window'):
                    continue
                try:
                    options[k] = eval(v)
                except NameError:
                    options[k] = v.rstrip()
        return options

    @staticmethod
    def _get_corr_indices(s1: str, s2: str) -> Tuple[int, int]:
        # s1 and s2 are one of 'Y', 'X1', 'X2', 'X3', ...
        i = 0 if s1 == 'Y' else int(s1[1:])
        j = 0 if s2 == 'Y' else int(s2[1:])
        return i, j

    def _update_correlations(self, num_x_vars: int, npts: int) -> None:
        # Performs the following:
        #   a) updates the _is_corr_array
        #   b) writes all in-memory correlation coefficient arrays to a file

        # always reset correlations from a previous call to fit()
        num_corr = num_x_vars + 1  # include y-variable
        self._is_corr_array[:num_corr, :num_corr] = False

        # update from files
        if self._corr_dir and self._corr_dir != self._tmp_dir:
            for filename in os.listdir(self._corr_dir):
                match = _corr_file_regex.match(filename)
                if match:
                    i, j = self._get_corr_indices(*match.groups())
                    self._is_corr_array[i, j] = True

        # update from memory
        for k, v in self._corr_dict.items():
            self._is_corr_array[v['index']] = True
            corr = v['corr']
            if isinstance(corr, float):
                corr = np.full((npts, npts), corr)
                np.fill_diagonal(corr, 1.0)
            elif corr.shape != (npts, npts):
                raise ValueError(f'Invalid {k!r} correlation array shape '
                                 f'[{corr.shape} != ({npts}, {npts})]')
            f = os.path.join(self._tmp_dir, f'CorrCoeffs {k}.txt')
            np.savetxt(f, corr, delimiter='\t')

    @staticmethod
    def create_parameters(parameters: Iterable[InputParameterType] = None) -> InputParameters:
        """Create a new collection of :class:`~msl.nlf.parameter.InputParameter`\\s.

        Parameters
        ----------
        parameters
            An iterable of either :class:`~msl.nlf.parameter.InputParameter`
            instances or objects that can be used to create an
            :class:`~msl.nlf.parameter.InputParameter` instance. See
            :meth:`~msl.nlf.parameter.InputParameters.add_many` for examples.
            If not specified, an empty collection is returned.

        Returns
        -------
        :class:`~msl.nlf.parameter.InputParameters`
            The input parameters.
        """
        return InputParameters(parameters)

    @property
    def dll_path(self) -> str:
        """Returns the path to the DLL file."""
        return self._dll_path

    @property
    def equation(self) -> str:
        """Returns the fitting equation."""
        return self._equation

    def evaluate(self,
                 x: ArrayLike,
                 result: Union[Result, Dict[str, float]]) -> np.ndarray[float]:
        """Evaluate the model to get the *y* (response) values.

        .. attention::

           This method calls the built-in :func:`eval` function which, in a
           malicious environment, can be used to execute arbitrary code. If
           you trust the :attr:`.equation` of this :class:`.Model`, there
           is no reason to be concerned.

        Parameters
        ----------
        x
            The independent variable (stimulus) data to evaluate the model at.
            If the model requires multiple variables, the *x* array must have
            a shape of *(num variables, num points)*, i.e., the data for each
            variable is listed per row

                .. centered::
                    [ [data for x1], [data for x2], ... ]

        result
            The fit result or a mapping between parameter names and values,
            e.g., *{'a1': 9.51, 'a2': -0.076, 'a3': 0.407}*.

        Returns
        -------
        :class:`~numpy.ndarray`
            The *y* (response) values.
        """
        equation = self._equation.replace('^', '**')

        try:
            namespace = dict((p.name, p.value) for p in result.params)
        except AttributeError:
            namespace = result

        namespace.update(**self._np_map)

        x = np.asarray(x)
        if x.ndim == 1:
            nvars, npts = (1, x.size)
        elif x.ndim == 2:
            nvars, npts = x.shape
        else:
            raise ValueError(f'Invalid shape of x data, {x.shape}')

        if nvars != self._num_vars:
            raise ValueError(f'Unexpected number of x variables '
                             f'[{nvars} != {self._num_vars}]')

        if nvars == 1:
            namespace['x'] = x
        else:
            namespace['x'] = x[0]  # x1 can also be written as x
            for i, row in enumerate(x, start=1):
                namespace[f'x{i}'] = row

        return eval(equation, {}, namespace)

    @overload
    def fit(self,
            x: ArrayLike,
            y: ArrayLike1D,
            *,
            params: Union[ArrayLike1D, InputParameters] = None,
            ux: ArrayLike = None,
            uy: ArrayLike1D = None,
            debug: Literal[False] = False,
            **options) -> Result: ...
    # Type hint: returns Result when debug=False (default)

    @overload
    def fit(self,
            x: ArrayLike,
            y: ArrayLike1D,
            *,
            params: Union[ArrayLike1D, InputParameters] = None,
            ux: ArrayLike = None,
            uy: ArrayLike1D = None,
            debug: Literal[True],
            **options) -> Input: ...
    # Type hint: returns Input when debug=True

    def fit(self,
            x: ArrayLike,
            y: ArrayLike1D,
            *,
            params: Union[ArrayLike1D, InputParameters] = None,
            ux: ArrayLike = None,
            uy: ArrayLike1D = None,
            debug: bool = False,
            **options):
        """Fit the model to the data.

        .. tip::

            It is more efficient to use an :class:`~numpy.ndarray` rather than
            a :class:`list` for the *x*, *y*, *ux* and *uy* arrays.

        Parameters
        ----------
        x
            The independent variable (stimulus) data. If the model requires
            multiple variables, the *x* array must have a shape of
            *(num variables, num points)*, i.e., the data for each variable
            is listed per row

            .. centered::
                [ [data for x1], [data for x2], ... ]
        y
            The dependent variable (response) data.
        params
            Fit parameters. If an array is passed in then every parameter will
            be allowed to vary during the fit. If you want more control, pass
            in an :class:`~msl.nlf.parameter.InputParameters` instance. If not
            specified, then the parameters are chosen from the :meth:`.guess`
            method.
        ux
            Standard uncertainties in the x data.
        uy
            Standard uncertainties in the y data.
        debug
            If enabled, a summary of the input data that would be passed to the
            fitting model is returned (the DLL is not called). Enabling this
            parameter is useful for debugging issues if the DLL raises an error
            or if the fit result is unexpected (e.g., the data points with
            smaller uncertainties are not having a stronger influence on the
            result, perhaps because an unweighted fit has been selected as one
            of the fit *options*).
        **options
            All additional keyword arguments are passed to :meth:`.options`.

        Returns
        -------
        :class:`.Result` or :class:`.Input`
            The returned type depends on whether *debug* mode is enabled or
            disabled. If *debug* is :data:`True` then an :class:`.Input` object
            is returned, otherwise a :class:`.Result` object is returned.
        """
        x = _fill_array(self._x, x)
        nvars, npts = (1, x.size) if x.ndim == 1 else x.shape
        if nvars != self._num_vars:
            raise ValueError(f'Unexpected number of x (stimulus) variables '
                             f'[{nvars} != {self._num_vars}]')

        y = _fill_array(self._y, y)
        if len(y) != npts:
            raise ValueError(f'len(y) != len(x) [{len(y)} != {npts}]')

        if params is None:
            params = self.guess(x, y)

        nparams = len(params)
        if isinstance(params, InputParameters):
            parameters = params
        else:
            parameters = self.create_parameters()
            for i, value in enumerate(params, start=1):
                parameters[f'a{i}'] = value

        _fill_array(self._a, parameters.values())
        _fill_array(self._constant, parameters.constants())
        if nparams != self._num_params:
            raise ValueError(f'Unexpected number of parameters '
                             f'[{nparams} != {self._num_params}]')

        ux = _fill_array(self._ux, ux)
        if (ux is not None) and (x.shape != ux.shape):
            raise ValueError(f'x.shape != ux.shape [{x.shape} != {ux.shape}]')

        uy = _fill_array(self._uy, uy)
        if (uy is not None) and (len(y) != len(uy)):
            raise ValueError(f'len(y) != len(uy) [{len(y)} != {len(uy)}]')

        if options:
            self.options(**options)

        self._update_correlations(nvars, npts)

        if self._show_warnings:
            if not self._weighted and (ux is not None or uy is not None):
                warnings.warn(
                    'uncertainties are specified but the fit option is set to unweighted',
                    UserWarning,
                    stacklevel=2
                )

        if debug:
            info = self._load_options()
            info.update(
                correlated=self._correlated,
                correlations=self._load_correlations(nvars),
                equation=self._equation,
                x=self._x[:nvars, :npts],
                y=self._y[:npts],
                params=parameters,
                ux=self._ux[:nvars, :npts],
                uy=self._uy[:npts]
            )
            return Input(**info)

        kwargs = {
            'cfg_path': self._cfg_path,
            'equation': self._equation,
            'x': self._x,
            'y': self._y,
            'ux': self._ux,
            'uy': self._uy,
            'npts': npts,
            'nvars': nvars,
            'a': self._a,
            'constant': self._constant,
            'correlated': self._correlated,
            'is_corr_array': self._is_corr_array,
            'corr_dir': self._corr_dir,
            'nparams': nparams,
            'max_iterations': self._max_iterations,
        }

        if isinstance(self._dll, ClientNLF):
            result = self._dll.fit(**kwargs)
        else:
            result = fit(self._dll.lib, covar=self._covar, ua=self._ua, **kwargs)

        result['dof'] = npts - nparams + int(np.sum(self._constant[:nparams]))
        result['params'] = ResultParameters(result, parameters)

        # calculate correlation matrix from covariance matrix
        # corr = cov(i,j)/sqrt(cov(i,i)*cov(j,j))
        corr = result['covariance'].copy()
        with warnings.catch_warnings():
            # ignore "RuntimeWarning: invalid value encountered in divide"
            warnings.simplefilter('ignore')
            stddev = np.sqrt(np.diag(corr))
            corr /= stddev[:, None]
            corr /= stddev[None, :]
        corr[result['covariance'] == 0] = 0
        result['correlation'] = corr

        return Result(**result)

    def guess(self, x: ArrayLike, y: ArrayLike1D, **kwargs) -> InputParameters:
        """Generate an initial guess for the parameters of a :class:`.Model`.

        Parameters
        ----------
        x
            The independent variable (stimulus) data. If the model requires
            multiple variables, the *x* array must have a shape of
            *(num variables, num points)*, i.e., the data for each variable
            is listed per row

            .. centered::
                [ [data for x1], [data for x2], ... ]
        y
            The dependent variable (response) data.
        **kwargs
            All additional keyword arguments are passed to the :class:`.Model`
            subclass.

        Returns
        -------
        :class:`~msl.nlf.parameter.InputParameters`
            Initial guesses for the parameters of a :class:`.Model`. Each
            :attr:`~msl.nlf.parameter.InputParameter.constant` value is set to
            :data:`False` and a :attr:`~msl.nlf.parameter.Parameter.label` is
            chosen. The values of these attributes can be changed by the user
            in the returned :class:`~msl.nlf.parameter.InputParameters` object.

        Raises
        ------
        NotImplementedError
            If the *guess* method is not implemented.
        """
        raise NotImplementedError(f'{self.__class__.__name__!r} has not '
                                  f'implemented a guess() method')

    @property
    def num_parameters(self) -> int:
        """Returns the number of fitting parameters in the equation."""
        return self._num_params

    @property
    def num_variables(self) -> int:
        """Returns the number of *x* (stimulus) variables in the equation."""
        return self._num_vars

    def options(self,
                *,
                correlated: bool = False,
                delta: float = 0.1,
                max_iterations: int = 999,
                method: Union[FitMethod, str] = FitMethod.LM,
                second_derivatives: bool = True,
                tolerance: float = 1e-20,
                weighted: bool = False,
                y_uncertainties_only: bool = False) -> None:
        """Configure the fitting options.

        Parameters
        ----------
        correlated
            Whether to include the correlations in the fitting process. Including
            correlations in the fit is only possible for least-squares fitting,
            in which case the fit becomes a generalised least-squares fit. The
            correlations between the correlated variables can be set by calling
            :meth:`.set_correlation` or :meth:`.set_correlation_dir`.
        delta
            Only used for Amoeba fitting.
        max_iterations
            The maximum number of fit iterations allowed.
        method
            The fitting method to use. Can be a member name or value of the
            :class:`~msl.nlf.datatypes.FitMethod` enum.
        second_derivatives
            Whether second-derivative terms (e.g., Hessian) are included in the
            propagation of uncertainty calculations.
        tolerance
            The fitting process will stop when the relative change in chi-square
            (or some other appropriate measure) is less than this value.
        weighted
             Whether to include the standard uncertainties in the fitting process
             to perform a weighted fit.
        y_uncertainties_only
            Whether the *y* uncertainties only or a combination of the *x* and *y*
            uncertainties are used to calculate the weights for a weighted fit.
        """
        # For details on how to create the file, see the ReadConfigFile function in
        # https://github.com/MSLNZ/Nonlinear-Fitting/blob/main/NLF%20DLL/NLFDLL.dpr
        # NOTE: Since Nonlinear-Fitting is a private repository, you must be logged
        #       in to GitHub to view the source code.
        #
        # AbsoluteRes and ResidualType are not kwargs because residuals are not
        # returned from the DLL.
        #
        # ShowInfoWindow is not a kwarg because the popup Window flashes to quickly
        # to be useful.
        self._weighted = bool(weighted)
        self._correlated = bool(correlated)
        self._max_iterations = max_iterations

        if not isinstance(method, FitMethod):
            try:
                method = FitMethod(method)
            except ValueError:
                try:
                    method = FitMethod[method]
                except KeyError:
                    raise ValueError(f'{method!r} is not a valid FitMethod '
                                     f'enum member name or value') from None

        with open(self._cfg_path, mode='wt') as f:
            f.write(f'weighted={bool(weighted)}\n'  # S='True';
                    f'max_iterations={max_iterations}\n'  # StrToInt(S);
                    f'tolerance={tolerance}\n'  # StrToFloat(S);
                    f'delta={delta}\n'  # StrToFloat(S);
                    f'absolute_res=Absolute\n'  # S='Absolute';
                    f'residual_type=dy v x\n'  # S=one of 'dx v x', 'dy v x', 'dx v y', 'dy v y'
                    f'fitting_method={method.value}\n'  # S=one of FittingMethod values
                    f'second_derivs_H={bool(second_derivatives)}\n'  # S='True';
                    f'second_derivs_B={bool(second_derivatives)}\n'  # S='True';
                    f'uy_weights_only={bool(y_uncertainties_only)}\n'  # S='True';
                    f'show_info_window=False')  # S='True';

    def remove_correlations(self) -> None:
        """Set all variables to be uncorrelated."""
        self._corr_dir = ''
        self._corr_dict.clear()

        for filename in os.listdir(self._tmp_dir):
            match = _corr_file_regex.match(filename)
            if match:
                os.remove(os.path.join(self._tmp_dir, filename))

    def set_correlation(self,
                        n1: str,
                        n2: str,
                        *,
                        matrix: ArrayLike = None,
                        value: float = None) -> None:
        """Set the correlation coefficients for the correlated variables.

        Note that the *x1-x2* correlation coefficients are identically equal
        to the *x2-x1* correlation coefficients, so only one of these relations
        needs to be defined.

        .. warning::

           It is recommended to not call :meth:`.set_correlation` and
           :meth:`.set_correlation_dir` with the same :class:`.Model` instance.
           Pick only one method. If you set correlations using both methods an
           error will *not* be raised, but you *may* be surprised which
           correlations are used.

        Parameters
        ----------
        n1
            The name of the first correlated variable (e.g., *y*, *x*, *x1*, *x2*).
        n2
            The name of the second correlated variable.
        matrix
            The coefficients of the correlation matrix.
        value
            Set all off-diagonal correlation coefficients to this value.
        """
        if value is None and matrix is None:
            raise ValueError("Specify either 'value' or 'matrix'")
        if value is not None and matrix is not None:
            raise ValueError("Cannot specify both 'value' and 'matrix'")

        s1, s2 = n1.upper(), n2.upper()
        if s1 == 'X':
            s1 = 'X1'
        if s2 == 'X':
            s2 = 'X1'

        def _check_var_name(s, n):
            match = _corr_var_regex.match(s)
            if not match:
                raise ValueError(f'Invalid correlation variable name {n!r}')
            name = match.group(1)[1:]
            if name and not (1 <= int(name) <= self._num_vars):
                raise ValueError(f'Invalid correlation variable name {n!r}, '
                                 f'variable X index outside of range')

        _check_var_name(s1, n1)
        _check_var_name(s2, n2)

        if value is not None:
            corr = float(value)
        else:
            corr = np.asarray(matrix)  # noqa: matrix cannot be None
            if corr.ndim != 2:
                raise ValueError(f'Invalid correlation matrix dimension '
                                 f'[{corr.ndim} != 2]')

        i, j = self._get_corr_indices(s1, s2)
        self._corr_dict[f'{s1}-{s2}'] = {'index': (i, j), 'corr': corr}
        self._corr_dir = self._tmp_dir

    def set_correlation_dir(self, directory: Union[str, None]) -> None:
        """Set the directory where the correlation coefficients are located.

        The directory should contain correlation-coefficient files that must
        be named *CorrCoeffs Y-Y.txt*, *CorrCoeffs X1-X1.txt*,
        *CorrCoeffs X1-X2.txt*, etc. Note that the *X1-X2* correlation
        coefficients are identically equal to the *X2-X1* correlation
        coefficients, so only one of the files *CorrCoeffs X1-X2.txt* or
        *CorrCoeffs X2-X1.txt* needs to be created.

        .. warning::

           It is recommended to not call :meth:`.set_correlation` and
           :meth:`.set_correlation_dir` with the same :class:`.Model` instance.
           Pick only one method. If you set correlations using both methods an
           error will *not* be raised, but you *may* be surprised which
           correlations are used.

        Parameters
        ----------
        directory
            The directory (folder) where the correlation coefficients are located.
            Specify ``.`` for the current working directory.
        """
        if not directory:
            self._corr_dir = ''  # must always be a string
            return

        if not os.path.isdir(directory):
            raise OSError(f'{directory!r} is not a valid directory')
        self._corr_dir = directory

    @property
    def show_warnings(self) -> bool:
        """Whether warning messages are shown."""
        return self._show_warnings

    @show_warnings.setter
    def show_warnings(self, show):
        self._show_warnings = bool(show)

    def version(self) -> str:
        """Get the version number from the DLL.

        The *GetVersion* function was added to the DLL in v5.41. If an older
        version of the DLL is loaded when a :class:`.Model` is created,
        calling this method will raise an exception.

        Returns
        -------
        str
            The version of the DLL.
        """
        if self._version:
            return self._version

        if isinstance(self._dll, ClientNLF):
            ver = self._dll.dll_version()
        else:
            ver = version(self._dll.lib)

        self._version = ver
        return ver


class CompositeModel(Model):

    def __init__(self,
                 op: str,
                 left: Model,
                 right: Model,
                 **kwargs) -> None:
        """Combine two models.

        Parameters
        ----------
        op
            A binary operator: ``+ - * /``.
        left
            The model on the left side of the operator.
        right
            The model on the right side of the operator.
        **kwargs
            All keyword arguments are passed to :class:`.Model`.
        """
        if op not in ('+', '-', '*', '/'):
            raise ValueError(f'Unsupported operator {op!r}')

        rhs = right.equation
        lhs = left.equation

        # remove the scaling factor or the offset from the right equation if it
        # also appears in the left equation for the appropriate binary operator
        index_offset = 0
        lhs_params = _n_params_regex.findall(lhs)
        rhs_params = _n_params_regex.findall(rhs)
        if (op in '*/') and \
                (left._factor in lhs_params and right._factor in rhs_params):
            rhs = right._composite_equation
            index_offset = 1
        elif (op in '+-') and \
                (left._offset in lhs_params and right._offset in rhs_params):
            rhs = right._composite_equation
            index_offset = 1

        # increment the parameter index in the right equation based on
        # how many parameters there are in the left equation
        sub = []
        i, end, n = 0, 0, left.num_parameters
        for match in _n_params_regex.finditer(rhs):
            start, end = match.span()
            new_index = int(match.group()[1:]) + n - index_offset
            sub.append(f'{rhs[i:start]}a{new_index}')
            i = end
        sub.append(rhs[end:])
        rhs = ''.join(sub)

        equation = f'({lhs}){op}({rhs})' if rhs else lhs
        super().__init__(equation, **kwargs)
