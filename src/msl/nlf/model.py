"""A model to use for a non-linear fit."""

from __future__ import annotations

import re
import sys
import warnings
from array import array
from ctypes import POINTER, c_double
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import TYPE_CHECKING, overload

import numpy as np

from msl.loadlib import LoadLibrary  # type: ignore[import-untyped]

from .client_server import ClientNLF
from .datatypes import Correlation, Correlations, FitMethod, Input, ResidualType, Result
from .dll import NPAR, NPTS, NVAR, define_fit_fcn, evaluate, fit, get_user_defined, version
from .parameter import InputParameters, ResultParameters
from .saver import save

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Any, Iterable, Literal, TypeVar

    from numpy.typing import ArrayLike, NDArray

    if sys.version_info[:2] < (3, 11):
        Self = TypeVar("Self", bound="Model")
    else:
        from typing import Self

    from .types import ArrayLike1D, ArrayLike2D, CorrDict, GetFunctionValue, InputParameterType


_n_params_regex = re.compile(r"a\d+")
_n_vars_regex = re.compile(r"(?<!e)(x\d*)")
_corr_var_regex = re.compile(r"^(Y|X\d{1,2})$")
_corr_file_regex = re.compile(r"^CorrCoeffs (Y|X\d{1,2})-(Y|X\d{1,2})\.txt$")
_user_fcn_regex = re.compile(r"^\s*f\d+\s*$")

_winreg_user_dir: str = ""

IS_PYTHON_64BIT: bool = sys.maxsize > 2**32


def _fill_array(a: NDArray[np.float64], b: ArrayLike) -> NDArray[np.float64]:
    """Fill array *a* with the values in array *b*. Returns *b*."""
    b = np.asanyarray(b, dtype=np.float64)
    if b.ndim == 1:
        if a.ndim == 1:
            a[: b.size] = b
        else:
            a[0, : b.size] = b
    elif b.ndim == 2:  # noqa: PLR2004
        i, j = b.shape
        a[:i, :j] = b
    else:
        msg = f"An array of shape {b.shape} is not supported"
        raise ValueError(msg)
    return b


def _get_user_dir() -> Path:
    """Get the user directory from the Windows registry.

    Reads the Windows Registry to get the value of the "User-defined function DLL directory"
    that is set by the Delphi GUI.

    If the Registry value does not exist, the current working directory
    is returned.
    """
    global _winreg_user_dir  # noqa: PLW0603
    if _winreg_user_dir:
        return Path(_winreg_user_dir)

    try:
        import winreg
    except ModuleNotFoundError:
        return Path.cwd().absolute()

    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"SOFTWARE\Measurement Standards Laboratory\Nonlinear Fitting\File Directories"
        )
    except OSError:
        pass
    else:
        index = 0
        while True:
            try:
                name, value, _ = winreg.EnumValue(key, index)
            except OSError:  # noqa: PERF203
                # No more data is available or didn't find "User-defined function DLL directory"
                break
            else:
                if name.startswith("User-defined"):
                    _winreg_user_dir = value.rstrip("\\")
                    break
                index += 1
        winreg.CloseKey(key)
    return Path(_winreg_user_dir)


_np_map = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "ln": np.log,
    "log": np.log10,
    "arcsin": np.arcsin,
    "arcos": np.arccos,
}


class Model:
    """A model for non-linear fitting."""

    MAX_POINTS: int = NPTS
    """Maximum number of data points allowed."""

    MAX_PARAMETERS: int = NPAR
    """Maximum number of fit parameters allowed."""

    MAX_VARIABLES: int = NVAR
    """Maximum number of x (stimulus) variables allowed."""

    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        equation: str,
        *,
        dll: str | Path | None = None,
        user_dir: str | Path | None = None,
        **options: Any,  # noqa: ANN401
    ) -> None:
        """A model for non-linear fitting.

        Parameters
        ----------
        equation
            The fit equation. The x variables (stimulus) must be specified as
            *x1*, *x2*, etc. and the parameters as *a1*, *a2*, etc. If only
            one x-variable is required, it can be simply entered as x. The
            arithmetic operations and functions that are recognised are:

            .. centered::
                + - * / ^ sin cos tan exp ln log arcsin arcos

            where **^** indicates raising to the power. All white space is
            ignored in the equation. For example, to fit a general quadratic
            equation one would use ``a1+a2*x+a3*x^2``. The **sqrt** function
            can be written as **^0.5**, for example, **sqrt(2*x)** would be
            expressed as **(2*x)^0.5** in the equation.

            |

            If using a user-defined function that has been compiled to a DLL,
            the *equation* name must begin with *f* followed by a positive integer,
            for example, ``f1``. The *user_dir* keyword argument may also need
            to be set. See :ref:`nlf-user-defined-function`.
        dll
            The path to a non-linear fit DLL file. A default DLL is chosen
            based on the bitness of the Python interpreter. If you want to
            load a 32-bit DLL in 64-bit Python then set *dll* to be **nlf32**.
            See :ref:`nlf-32vs64` for reasons why you may want to use a
            different DLL bitness. You may also specify a path to a DLL that
            is located in a particular directory of your computer.
        user_dir
            Directory where the user-defined functions are located. The default
            directory is the directory that the Delphi GUI has set. If the
            Delphi GUI has not set a directory (because the GUI has not been
            used) the default directory is the current working directory.
            See :ref:`nlf-user-defined-function`.
        **options
            All additional keyword arguments are passed to :meth:`.options`.
        """
        self._dll: LoadLibrary | ClientNLF | None = None
        self._tmp_dir = Path(mkdtemp(prefix="nlf-"))
        self._cfg_path = self._tmp_dir / "options.cfg"
        self._equation = equation
        self._version = ""
        self._corr_dir = ""
        self._corr_dict: dict[tuple[int, int], CorrDict] = {}
        self._show_warnings = True
        self._parameters = InputParameters()
        self._npts = -1

        # fit options
        self._absolute_residuals = True
        self._correlated = False
        self._delta = 0.1
        self._max_iterations = 999
        self._fit_method = FitMethod.LM
        self._residual_type = ResidualType.DY_X
        self._second_derivs_B = True
        self._second_derivs_H = True
        self._tolerance = 1e-20
        self._uy_weights_only = False
        self._weighted = False

        # these are re-defined in Model subclasses and are
        # used when creating a CompositeModel
        self._factor = ""
        self._offset = ""
        self._composite_equation = ""

        if user_dir is None:
            user_dir = _get_user_dir()
        self._user_dir = Path(user_dir)

        self._user_function: GetFunctionValue | None = None
        self._user_function_name: str = ""
        self._is_user_function: bool = _user_fcn_regex.match(equation) is not None

        variables = set(_n_vars_regex.findall(equation))
        if "x" in variables and "x1" in variables:
            msg = "Cannot use both 'x' and 'x1' in equation"
            raise ValueError(msg)
        if "x0" in variables:
            msg = "Cannot use 'x0' in equation"
            raise ValueError(msg)

        self._num_vars: int = len(variables)
        if self._num_vars > self.MAX_VARIABLES:
            msg = f"Too many x variables in equation [{self._num_vars} > {self.MAX_VARIABLES}]"
            raise ValueError(msg)

        params = set(_n_params_regex.findall(equation))
        if "a0" in params:
            msg = "Cannot use 'a0' in equation"
            raise ValueError(msg)

        self._num_params: int = len(params)
        if self._num_params > self.MAX_PARAMETERS:
            msg = f"Too many fitting parameters in equation [{self._num_params} > {self.MAX_PARAMETERS}]"
            raise ValueError(msg)

        self.options(**options)

        self._x = np.zeros((self.MAX_VARIABLES, self.MAX_POINTS))
        self._ux = np.zeros((self.MAX_VARIABLES, self.MAX_POINTS))
        self._y = np.zeros(self.MAX_POINTS)
        self._uy = np.zeros(self.MAX_POINTS)
        self._a = np.zeros(self.MAX_PARAMETERS)
        self._constant = np.zeros(self.MAX_PARAMETERS, dtype=bool)
        self._is_corr_array = np.zeros((self.MAX_VARIABLES + 1, self.MAX_VARIABLES + 1), dtype=bool)

        here = Path(__file__).parent
        if not dll:
            bit = "64" if IS_PYTHON_64BIT else "32"
            dll = here / f"nlf{bit}.dll"
        elif dll == "nlf32":
            dll = here / "nlf32.dll"
        elif dll == "nlf64":
            if not IS_PYTHON_64BIT:
                msg = "Cannot load a 64-bit DLL in 32-bit Python"
                raise ValueError(msg)
            dll = here / "nlf64.dll"
        self._dll_path = Path(dll)

        try:
            self._dll = LoadLibrary(dll, libtype="cdll")
        except OSError as e:
            if e.winerror == 193:  # noqa: PLR2004
                # Tried to load a 32-bit DLL in 64-bit Python.
                # Use interprocess communication (see msl-loadlib).
                self._dll = ClientNLF(dll)
            else:
                raise
        else:
            self._covar = np.zeros((self.MAX_PARAMETERS, self.MAX_PARAMETERS))
            self._ua = np.zeros(self.MAX_PARAMETERS)
            define_fit_fcn(self._dll.lib, as_ctypes=False)

        if self._is_user_function:
            # this must come at the end since self._dll must not be None
            # and self._num_vars, self._num_params get overwritten
            self._load_user_defined()

    def __add__(self, rhs: Model | float) -> Model:
        """Override the + operator."""
        op = "+"
        if isinstance(rhs, Model):
            return CompositeModel(op, self, rhs, dll=self._dll_path)
        if isinstance(rhs, (float, int)):
            return Model(f"{self._equation}{op}{rhs}", dll=self._dll_path)
        msg = f"unsupported operand type(s) for {op}: {type(self).__name__!r} and {type(rhs).__name__!r}"
        raise TypeError(msg)

    def __radd__(self, lhs: float) -> Model:
        """Override the + operator."""
        op = "+"
        if isinstance(lhs, (float, int)):
            return Model(f"{lhs}{op}{self._equation}", dll=self._dll_path)
        msg = f"unsupported operand type(s) for {op}: {type(lhs).__name__!r} and {type(self).__name__!r}"
        raise TypeError(msg)

    def __sub__(self, rhs: Model | float) -> Model:
        """Override the - operator."""
        op = "-"
        if isinstance(rhs, Model):
            return CompositeModel(op, self, rhs, dll=self._dll_path)
        if isinstance(rhs, (float, int)):
            return Model(f"{self._equation}{op}{rhs}", dll=self._dll_path)
        msg = f"unsupported operand type(s) for {op}: {type(self).__name__!r} and {type(rhs).__name__!r}"
        raise TypeError(msg)

    def __rsub__(self, lhs: float) -> Model:
        """Override the - operator."""
        op = "-"
        if isinstance(lhs, (float, int)):
            return Model(f"{lhs}{op}{self._equation}", dll=self._dll_path)
        msg = f"unsupported operand type(s) for {op}: {type(lhs).__name__!r} and {type(self).__name__!r}"
        raise TypeError(msg)

    def __mul__(self, rhs: Model | float) -> Model:
        """Override the * operator."""
        op = "*"
        if isinstance(rhs, Model):
            return CompositeModel(op, self, rhs, dll=self._dll_path)
        if isinstance(rhs, (float, int)):
            return Model(f"({self._equation}){op}{rhs}", dll=self._dll_path)
        msg = f"unsupported operand type(s) for {op}: {type(self).__name__!r} and {type(rhs).__name__!r}"
        raise TypeError(msg)

    def __rmul__(self, lhs: float) -> Model:
        """Override the * operator."""
        op = "*"
        if isinstance(lhs, (float, int)):
            return Model(f"{lhs}{op}({self._equation})", dll=self._dll_path)
        msg = f"unsupported operand type(s) for {op}: {type(lhs).__name__!r} and {type(self).__name__!r}"
        raise TypeError(msg)

    def __truediv__(self, rhs: Model | float) -> Model:
        """Override the / operator."""
        op = "/"
        if isinstance(rhs, Model):
            return CompositeModel(op, self, rhs, dll=self._dll_path)
        if isinstance(rhs, (float, int)):
            return Model(f"({self._equation}){op}{rhs}", dll=self._dll_path)
        msg = f"unsupported operand type(s) for {op}: {type(self).__name__!r} and {type(rhs).__name__!r}"
        raise TypeError(msg)

    def __rtruediv__(self, lhs: float) -> Model:
        """Override the / operator."""
        op = "/"
        if isinstance(lhs, (float, int)):
            return Model(f"{lhs}{op}({self._equation})", dll=self._dll_path)
        msg = f"unsupported operand type(s) for {op}: {type(lhs).__name__!r} and {type(self).__name__!r}"
        raise TypeError(msg)

    def __del__(self) -> None:
        """Clean up when the reference count reaches zero."""
        try:  # noqa: SIM105
            self._cleanup()
        except AttributeError:
            pass

    def __enter__(self: Self) -> Self:
        """Enter a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit a context manager."""
        self._cleanup()

    def __repr__(self) -> str:
        """Object representation."""
        path = self._dll_path.name if Path(__file__).parent == self._dll_path.parent else str(self._dll_path)
        return f"{self.__class__.__name__}(equation={self._equation!r}, dll={path!r})"

    def _cleanup(self) -> None:
        if isinstance(self._dll, ClientNLF):
            self._dll.shutdown_server32()
        rmtree(self._tmp_dir, ignore_errors=True)

    def _load_correlations(self, num_x_vars: int) -> Correlations:
        # load the correlation files
        num_corr = num_x_vars + 1  # include y-variable
        coeffs = []
        if self._corr_dir:
            corr_dir = Path(self._corr_dir)
            for filename in corr_dir.glob("*.txt"):
                match = _corr_file_regex.match(filename.name)
                if match:
                    path = corr_dir / filename
                    coeffs.append(Correlation(path=path, coefficients=np.loadtxt(path)))

        return Correlations(is_correlated=self._is_corr_array[:num_corr, :num_corr], data=coeffs)

    def _load_options(self) -> dict[str, Any]:
        # load the options.cfg file
        options = {}
        ignore = ("show_info_window", "user_dir")
        with self._cfg_path.open() as f:
            for line in f:
                k, v = line.split("=")
                if k in ignore:
                    continue
                try:
                    options[k] = eval(v)  # noqa: S307
                except (NameError, SyntaxError):
                    options[k] = v.rstrip()
                if k == "fit_method":
                    options[k] = FitMethod(options[k])
                elif k == "absolute_residuals":
                    options[k] = options[k] == "Absolute"
                elif k == "residual_type":
                    options[k] = ResidualType(options[k])
        return options

    def _load_user_defined(self) -> None:
        # load the user-defined functions
        if self._dll is None:
            return

        if not self._user_dir.is_dir():
            msg = f"The user-defined directory does not exist: {str(self._user_dir)!r}"
            raise FileNotFoundError(msg)

        if isinstance(self._dll, ClientNLF):
            functions = self._dll.get_user_defined(self._user_dir)
        else:
            functions = get_user_defined(self._user_dir)

        # find all user-defined DLLs that have the expected equation
        ud_matches = [ud for ud in functions.values() if ud.equation == self._equation]

        if not ud_matches:
            e = f"No user-defined function named {self._equation!r} is in {self._user_dir!r}"
            if functions:
                names = "\n  ".join({v.name for v in functions.values()})
                e += f"\nThe functions available are:\n  {names}"
            else:
                e += "\nThere are no valid functions in this directory."
            e += "\nMake sure that the bitness of the user-defined DLL and the bitness of the NLF DLL are the same."
            raise ValueError(e)

        if len(ud_matches) > 1:
            names = "\n  ".join(str(filename) for filename in sorted(functions))
            e = (
                f"Multiple user-defined functions named {self._equation!r} "
                f"were found in {self._user_dir!r}\n  {names}"
            )
            raise ValueError(e)

        ud = ud_matches[0]
        self._user_function_name = ud.name
        self._num_params = ud.num_parameters
        self._num_vars = ud.num_variables

        if isinstance(self._dll, ClientNLF):
            self._dll.load_user_defined(self._equation, self._user_dir)
        else:
            self._user_function = ud.function
            self._user_function.restype = None
            p = np.ctypeslib.ndpointer
            self._user_function.argtypes = [
                p(dtype=c_double, flags="C_CONTIGUOUS"),
                p(dtype=c_double, flags="C_CONTIGUOUS"),
                POINTER(c_double),
            ]

    @staticmethod
    def _get_corr_indices(s1: str, s2: str) -> tuple[int, int]:
        # s1 and s2 are one of 'Y', 'X1', 'X2', 'X3', ...
        i = 0 if s1 == "Y" else int(s1[1:])
        j = 0 if s2 == "Y" else int(s2[1:])
        return i, j

    def _update_correlations(self, num_x_vars: int, npts: int) -> None:
        # Performs the following:
        #   a) updates the _is_corr_array
        #   b) writes all in-memory correlation coefficient arrays to a file

        # always reset correlations from a previous call to fit()
        num_corr = num_x_vars + 1  # include y-variable
        self._is_corr_array[:num_corr, :num_corr] = False

        # update from files
        if self._corr_dir and self._corr_dir != str(self._tmp_dir):
            for filename in Path(self._corr_dir).glob("*.txt"):
                match = _corr_file_regex.match(filename.name)
                if match:
                    i, j = self._get_corr_indices(*match.groups())
                    self._is_corr_array[i, j] = True
                    self._is_corr_array[j, i] = True

        # update from memory
        for k, v in self._corr_dict.items():
            i, j = k
            self._is_corr_array[i, j] = True
            self._is_corr_array[j, i] = True
            corr = v["corr"]
            if isinstance(corr, float):
                corr = np.full((npts, npts), corr)
                np.fill_diagonal(corr, 1.0)
            elif corr.shape != (npts, npts):
                names = "-".join(v["names"])
                msg = f"Invalid {names!r} correlation array shape [{corr.shape} != ({npts}, {npts})]"
                raise ValueError(msg)
            n1, n2 = v["names"]
            f = self._tmp_dir / f"CorrCoeffs {n1}-{n2}.txt"
            np.savetxt(f, corr)
            if n1 != n2:
                f = self._tmp_dir / f"CorrCoeffs {n2}-{n1}.txt"
                np.savetxt(f, corr)

    @staticmethod
    def create_parameters(parameters: Iterable[InputParameterType] | None = None) -> InputParameters:
        r"""Create a new collection of :class:`~msl.nlf.parameter.InputParameter`\\s.

        Parameters
        ----------
        parameters
            An iterable of either :class:`~msl.nlf.parameter.InputParameter`
            instances or objects that can be used to create an
            :class:`~msl.nlf.parameter.InputParameter` instance. See
            :meth:`~msl.nlf.parameter.InputParameters.add_many` for examples.
            If not specified, an empty collection is returned.

        Returns:
        -------
        :class:`~msl.nlf.parameter.InputParameters`
            The input parameters.
        """
        return InputParameters(parameters)

    @property
    def dll_path(self) -> Path:
        """Returns the path to the DLL file."""
        return self._dll_path

    @property
    def equation(self) -> str:
        """Returns the fitting equation."""
        return self._equation

    def evaluate(self, x: ArrayLike1D | ArrayLike2D, result: Result | dict[str, float]) -> NDArray[np.float64]:
        """Evaluate the model to get the *y* (response) values.

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

        Returns:
        -------
        :class:`~numpy.ndarray`
            The *y* (response) values.
        """
        x = np.asanyarray(x)
        if x.ndim == 1:
            nvars, npts = (1, x.size)
        elif x.ndim == 2:  # noqa: PLR2004
            nvars, npts = x.shape
        else:
            msg = f"Invalid shape of x data, {x.shape}"
            raise ValueError(msg)

        if nvars != self._num_vars:
            msg = f"Unexpected number of x variables [{nvars} != {self._num_vars}]"
            raise ValueError(msg)

        if self._is_user_function:
            a = result.params.values() if isinstance(result, Result) else np.array(list(result.values()))

            shape = (nvars, npts)
            xtf = x.T.reshape(-1)  # transpose and flatten
            if isinstance(self._dll, ClientNLF):
                a_array = array("d", a.tobytes())
                x_array = array("d", xtf.tobytes())
                return np.array(self._dll.evaluate(a_array, x_array, shape))

            if self._user_function is None:
                msg = "self._user_function cannot be None"
                raise AssertionError(msg)

            y = np.empty(npts, dtype=float)
            return evaluate(self._user_function, a, xtf, shape, y)

        namespace = {p.name: p.value for p in result.params} if hasattr(result, "params") else result
        namespace.update(**_np_map)

        if nvars == 1:
            namespace["x"] = x
            namespace["x1"] = x  # x can also be written as x1
        else:
            namespace["x"] = x[0]  # x1 can also be written as x
            for i, row in enumerate(x, start=1):
                namespace[f"x{i}"] = row

        equation = self._equation.replace("^", "**")
        values: NDArray[np.float64] = eval(equation, {"__builtins__": {}}, namespace)  # noqa: S307
        return values

    @overload
    def fit(
        self,
        x: ArrayLike1D | ArrayLike2D,
        y: ArrayLike1D,
        *,
        params: ArrayLike1D | InputParameters | None = None,
        ux: ArrayLike1D | ArrayLike2D | None = None,
        uy: ArrayLike1D | None = None,
        debug: Literal[False] = False,
        **options: Any,  # noqa: ANN401
    ) -> Result: ...

    @overload
    def fit(
        self,
        x: ArrayLike1D | ArrayLike2D,
        y: ArrayLike1D,
        *,
        params: ArrayLike1D | InputParameters | None = None,
        ux: ArrayLike1D | ArrayLike2D | None = None,
        uy: ArrayLike1D | None = None,
        debug: Literal[True],
        **options: Any,  # noqa: ANN401
    ) -> Input: ...

    def fit(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        x: ArrayLike1D | ArrayLike2D,
        y: ArrayLike1D,
        *,
        params: ArrayLike1D | InputParameters | None = None,
        ux: ArrayLike1D | ArrayLike2D | None = None,
        uy: ArrayLike1D | None = None,
        debug: bool = False,
        **options: Any,
    ) -> Result | Input:
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
            fit function in the DLL is returned (the DLL function is not called).
            Enabling this parameter is useful for debugging issues if the DLL
            raises an error or if the fit result is unexpected (e.g., the data
            points with smaller uncertainties are not having a stronger influence
            on the result, perhaps because an unweighted fit has been selected
            as one of the fit *options*).
        **options
            All additional keyword arguments are passed to :meth:`.options`.

        Returns:
        -------
        :class:`.Result` or :class:`.Input`
            The returned type depends on whether *debug* mode is enabled or
            disabled. If *debug* is :data:`True` then an :class:`.Input` object
            is returned, otherwise a :class:`.Result` object is returned.
        """
        x = _fill_array(self._x, x)
        nvars, npts = (1, x.size) if x.ndim == 1 else x.shape
        if self._num_vars > 0 and self._num_vars != nvars:
            msg = f"Unexpected number of x (stimulus) variables [{nvars} != {self._num_vars}]"
            raise ValueError(msg)

        y = _fill_array(self._y, y)
        if len(y) != npts:
            msg = f"len(y) != len(x) [{len(y)} != {npts}]"
            raise ValueError(msg)

        if params is None:
            params = self.guess(x, y)

        nparams = len(params)
        if isinstance(params, InputParameters):
            parameters = params
        else:
            parameters = self.create_parameters()
            for i, value in enumerate(params, start=1):
                parameters[f"a{i}"] = value

        self._npts = npts
        self._parameters = parameters
        _fill_array(self._a, parameters.values())
        _fill_array(self._constant, parameters.constants())
        if nparams != self._num_params:
            msg = f"Unexpected number of parameters [{nparams} != {self._num_params}]"
            raise ValueError(msg)

        if ux is not None:
            ux = _fill_array(self._ux, ux)
            if (x.shape != ux.shape) and not (nvars == 1 and npts == ux.shape[0]):
                # allow a 1D ux array provided that nvars=1
                # for example, if x.shape=(1, 3) and ux.shape=(3,) then it is ok
                msg = f"x.shape != ux.shape [{x.shape} != {ux.shape}]"
                raise ValueError(msg)

        if uy is not None:
            uy = _fill_array(self._uy, uy)
            if len(y) != len(uy):
                msg = f"len(y) != len(uy) [{len(y)} != {len(uy)}]"
                raise ValueError(msg)

        if options:
            self.options(**options)

        self._update_correlations(nvars, npts)

        if self._show_warnings:
            if (not self._weighted) and (ux is not None or uy is not None):
                warnings.warn("unweighted fit but uncertainties are specified", UserWarning, stacklevel=2)

            any_correlations = np.any(self._is_corr_array)
            if self._correlated:
                if not any_correlations:
                    warnings.warn("correlated fit but there are no correlations", UserWarning, stacklevel=2)
            elif any_correlations:
                warnings.warn("uncorrelated fit but correlations are specified", UserWarning, stacklevel=2)

        if debug:
            info = self._load_options()
            info.update(
                {
                    "correlated": self._correlated,
                    "weighted": self._weighted,
                    "correlations": self._load_correlations(nvars),
                    "equation": self._equation,
                    "x": self._x[:nvars, :npts],
                    "y": self._y[:npts],
                    "params": parameters,
                    "ux": self._ux[:nvars, :npts],
                    "uy": self._uy[:npts],
                }
            )
            return Input(**info)

        kwargs = {
            "cfg_path": str(self._cfg_path),
            "equation": self._equation,
            "x": self._x,
            "y": self._y,
            "ux": self._ux,
            "uy": self._uy,
            "npts": npts,
            "nvars": nvars,
            "a": self._a,
            "constant": self._constant,
            "correlated": self._correlated,
            "is_corr_array": self._is_corr_array,
            "corr_dir": self._corr_dir,
            "nparams": nparams,
            "max_iterations": self._max_iterations,
            "weighted": self._weighted,
        }

        if isinstance(self._dll, ClientNLF):
            result = self._dll.fit(**kwargs)
        else:
            result = fit(lib=self._dll.lib, covar=self._covar, ua=self._ua, **kwargs)  # type: ignore[union-attr]

        if self._weighted or self._correlated:
            result["dof"] = float("inf")
        else:
            result["dof"] = float(npts - nparams + np.sum(self._constant[:nparams]))

        result["params"] = ResultParameters(result, parameters)

        # calculate correlation matrix from covariance matrix: corr = cov(i,j)/sqrt(cov(i,i)*cov(j,j))
        corr = result["covariance"].copy()
        with warnings.catch_warnings():
            # ignore "RuntimeWarning: invalid value encountered in divide"
            warnings.simplefilter("ignore")
            stddev = np.sqrt(np.diag(corr))
            corr /= stddev[:, None]
            corr /= stddev[None, :]
        corr[result["covariance"] == 0] = 0
        result["correlation"] = corr

        if self._show_warnings and result["iterations"] >= self._max_iterations:
            warnings.warn(
                f'maximum number of fit iterations exceeded [{result["iterations"]}]', UserWarning, stacklevel=2
            )

        return Result(**result)

    def guess(self, x: ArrayLike1D | ArrayLike2D, y: ArrayLike1D, **kwargs: Any) -> InputParameters:  # noqa: ANN401, ARG002
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

        Returns:
        -------
        :class:`~msl.nlf.parameter.InputParameters`
            Initial guesses for the parameters of a :class:`.Model`. Each
            :attr:`~msl.nlf.parameter.InputParameter.constant` value is set to
            :data:`False` and a :attr:`~msl.nlf.parameter.Parameter.label` is
            chosen. The values of these attributes can be changed by the user
            in the returned :class:`~msl.nlf.parameter.InputParameters` object.

        Raises:
        ------
        NotImplementedError
            If the *guess* method is not implemented.
        """
        msg = f"{self.__class__.__name__!r} has not implemented a guess() method"
        raise NotImplementedError(msg)

    @property
    def num_parameters(self) -> int:
        """Returns the number of fitting parameters in the equation."""
        return self._num_params

    @property
    def num_variables(self) -> int:
        """Returns the number of *x* (stimulus) variables in the equation."""
        return self._num_vars

    def options(  # noqa: C901, PLR0913
        self,
        *,
        absolute_residuals: bool | None = None,
        correlated: bool | None = None,
        delta: float | None = None,
        max_iterations: int | None = None,
        fit_method: FitMethod | str | None = None,
        residual_type: ResidualType | str | None = None,
        second_derivs_B: bool | None = None,  # noqa: N803
        second_derivs_H: bool | None = None,  # noqa: N803
        tolerance: float | None = None,
        uy_weights_only: bool | None = None,
        weighted: bool | None = None,
    ) -> None:
        """Configure the fitting options.

        Parameters
        ----------
        absolute_residuals
            Whether absolute residuals or relative residuals are used to evaluate
            the :attr:`~msl.nlf.datatypes.Result.eof`. Default: True (absolute).
        correlated
            Whether to include the correlations in the fitting process. Including
            correlations in the fit is only possible for least-squares fitting,
            in which case the fit becomes a generalised least-squares fit. The
            correlations between the correlated variables can be set by calling
            :meth:`.set_correlation` or :meth:`.set_correlation_dir`.
            Default: False.
        delta
            Only used for Amoeba fitting. Default: 0.1.
        max_iterations
            The maximum number of fit iterations allowed. Default: 999.
        fit_method
            The fitting method to use. Can be a member name or value of the
            :class:`~msl.nlf.datatypes.FitMethod` enum.
            Default: Levenberg-Marquardt.
        residual_type
            The residual type to use to evaluate the :attr:`~msl.nlf.datatypes.Result.eof`.
            Can be a member name or value of the :class:`~msl.nlf.datatypes.ResidualType`
            enum. Default: DY_X (uncertainty in :math:`y` versus :math:`x`).
        second_derivs_B
            Whether the second derivatives in the **B** matrix are included in
            the propagation of uncertainty calculations. Default: True.
        second_derivs_H
            Whether the second derivatives in the curvature matrix, **H**
            (Hessian), are included in the propagation of uncertainty calculations.
            Default: True.
        tolerance
            The fitting process will stop when the relative change in chi-square
            (or some other appropriate measure) is less than this value.
            Default: 1e-20.
        uy_weights_only
            Whether the *y* uncertainties only or a combination of the *x* and *y*
            uncertainties are used to calculate the weights for a weighted fit.
            Default: False.
        weighted
             Whether to include the standard uncertainties in the fitting process
             to perform a weighted fit. Default: False.
        """

        # For details on how to create the file, see the ReadConfigFile function in
        # https://github.com/MSLNZ/Nonlinear-Fitting/blob/main/NLF%20DLL/NLFDLL.dpr
        # NOTE: Since Nonlinear-Fitting is a private repository, you must be logged
        #       in to GitHub to view the source code.
        #
        # ShowInfoWindow is not a kwarg because the popup Window flashes to quickly
        # to be useful.
        def get_enum(item: Any, enum: Any) -> Any:  # noqa: ANN401
            if not isinstance(item, enum):
                try:
                    item = enum(item)
                except ValueError:
                    try:
                        item = enum[item]
                    except KeyError:
                        msg = f"{item!r} is not a valid {enum.__name__} enum member name or value"
                        raise ValueError(msg) from None
            return item

        if absolute_residuals is not None:
            self._absolute_residuals = bool(absolute_residuals)
        if correlated is not None:
            self._correlated = bool(correlated)
        if delta is not None:
            self._delta = float(delta)
        if max_iterations is not None:
            self._max_iterations = int(max_iterations)
        if second_derivs_B is not None:
            self._second_derivs_B = bool(second_derivs_B)
        if second_derivs_H is not None:
            self._second_derivs_H = bool(second_derivs_H)
        if tolerance is not None:
            self._tolerance = float(tolerance)
        if uy_weights_only is not None:
            self._uy_weights_only = bool(uy_weights_only)
        if weighted is not None:
            self._weighted = bool(weighted)
        if residual_type is not None:
            self._residual_type = get_enum(residual_type, ResidualType)
        if fit_method is not None:
            self._fit_method = get_enum(fit_method, FitMethod)

        abs_res = "Absolute" if self._absolute_residuals else "Relative"
        with self._cfg_path.open(mode="w") as f:
            f.write(
                f"max_iterations={self._max_iterations}\n"  # StrToInt(S);
                f"tolerance={self._tolerance}\n"  # StrToFloat(S);
                f"delta={self._delta}\n"  # StrToFloat(S);
                f"absolute_residuals={abs_res}\n"  # S='Absolute';
                f"residual_type={self._residual_type.value}\n"  # S=one of ResidualType values
                f"fit_method={self._fit_method.value}\n"  # S=one of FitMethod values
                f"second_derivs_H={self._second_derivs_H}\n"  # S='True';
                f"second_derivs_B={self._second_derivs_B}\n"  # S='True';
                f"uy_weights_only={self._uy_weights_only}\n"  # S='True';
                f"show_info_window=False\n"  # S='True';
                f"user_dir={self._user_dir}\\"
            )

    def remove_correlations(self) -> None:
        """Set all variables to be uncorrelated."""
        self._corr_dir = ""
        self._corr_dict.clear()
        for filename in self._tmp_dir.glob("*.txt"):
            if _corr_file_regex.match(filename.name):
                (self._tmp_dir / filename).unlink()

    def save(  # noqa: PLR0913
        self,
        path: str | Path,
        *,
        x: ArrayLike1D | ArrayLike2D | None = None,
        y: ArrayLike1D | None = None,
        params: ArrayLike1D | InputParameters | None = None,
        ux: ArrayLike1D | ArrayLike2D | None = None,
        uy: ArrayLike1D | None = None,
        comments: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Save a **.nlf** file.

        The file can be opened in the Delphi GUI application or loaded via
        the :func:`~msl.nlf.load` function.

        No information about the fit results are written to the file. If you
        are opening the file in the Delphi GUI, you must click the *Calculate*
        button to perform the fit and create the graphs.

        Parameters
        ----------
        path
            The path to save the file to. The file extension must be **.nlf**.
        x
            The independent variable (stimulus) data. If not specified, the
            data that was most recently passed to :meth:`.fit` or a previous
            call to :meth:`.save` is used.
        y
            The dependent variable (response) data. If not specified, the
            data that was most recently passed to :meth:`.fit` or a previous
            call to :meth:`.save` is used.
        params
            Fit parameters. If not specified, the parameters that were
            most recently passed to :meth:`.fit` or a previous
            call to :meth:`.save` are used. Since the Delphi GUI application
            does not use the :attr:`~msl.nlf.parameter.InputParameter.label`
            attribute, the *labels* are not saved and will be :data:`None`
            when the file is reloaded.
        ux
            Standard uncertainties in the x data. If not specified, the
            data that was most recently passed to :meth:`.fit` is used.
        uy
            Standard uncertainties in the y data. If not specified, the
            data that was most recently passed to :meth:`.fit` is used.
        comments
            Additional comments to add to the file. This text will appear in
            the *Comments* window in the Delphi GUI application.
        overwrite
            Whether to overwrite the file if it already exists. If the file
            exists, and this value is :data:`False` then an error is raised.
        """
        nvars, npts = self._num_vars, self._npts

        if x is None:
            if npts < 0:
                msg = "Must specify x data before saving"
                raise ValueError(msg)
            x = self._x[:nvars, :npts]

        if y is None:
            if npts < 0:
                msg = "Must specify y data before saving"
                raise ValueError(msg)
            y = self._y[:npts]

        if params is None:
            if not self._parameters:
                msg = "Must specify params before saving"
                raise ValueError(msg)
            params = self._parameters

        c = comments or ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self.fit(x, y, params=params, ux=ux, uy=uy, debug=True)
        save(path=path, comments=c, overwrite=overwrite, data=data)

    def set_correlation(
        self, n1: str, n2: str, *, matrix: ArrayLike2D | None = None, value: float | None = None
    ) -> None:
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
            msg = "Specify either 'value' or 'matrix'"
            raise ValueError(msg)
        if value is not None and matrix is not None:
            msg = "Cannot specify both 'value' and 'matrix'"
            raise ValueError(msg)

        s1, s2 = n1.upper(), n2.upper()
        if s1 == "X":
            s1 = "X1"
        if s2 == "X":
            s2 = "X1"

        def _check_var_name(s: str, n: str) -> None:
            match = _corr_var_regex.match(s)
            if not match:
                msg = f"Invalid correlation variable name {n!r}"
                raise ValueError(msg)
            name = match.group(1)[1:]
            if name and not (1 <= int(name) <= self._num_vars):
                msg = f"Invalid correlation variable name {n!r}, variable X index outside of range"
                raise ValueError(msg)

        _check_var_name(s1, n1)
        _check_var_name(s2, n2)

        corr: float | NDArray[np.float64]
        if value is not None:
            corr = float(value)
        else:
            corr = np.asanyarray(matrix)
            if corr.ndim != 2:  # noqa: PLR2004
                msg = f"Invalid correlation matrix dimension [{corr.ndim} != 2]"
                raise ValueError(msg)

        i, j = self._get_corr_indices(s1, s2)
        self._corr_dict[(i, j)] = {"corr": corr, "names": (s1, s2)}
        self._corr_dir = str(self._tmp_dir)

    def set_correlation_dir(self, directory: str | Path | None) -> None:
        """Set the directory where the correlation coefficients are located.

        The directory should contain correlation-coefficient files that must
        be named *CorrCoeffs Y-Y.txt*, *CorrCoeffs X1-X1.txt*,
        *CorrCoeffs X1-X2.txt*, etc. Note that the *X1-X2* correlation
        coefficients are identically equal to the *X2-X1* correlation
        coefficients, so only one of the files *CorrCoeffs X1-X2.txt* or
        *CorrCoeffs X2-X1.txt* needs to be created.

        Whitespace is used to separate the value for each column in a file.

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
            self._corr_dir = ""  # must always be a string
            return

        path = Path(directory)
        if not path.is_dir():
            msg = f"{str(path)!r} is not a valid directory"
            raise OSError(msg)

        self._corr_dir = str(path.absolute())

    @property
    def show_warnings(self) -> bool:
        """Whether warning messages are shown.

        Warnings are shown if correlations are defined and the fit option is
        set to be uncorrelated, or if *ux* or *uy* are specified and the fit
        option is unweighted, or if the maximum number of fit iterations
        has been exceeded.
        """
        return self._show_warnings

    @show_warnings.setter
    def show_warnings(self, show: bool) -> None:
        self._show_warnings = bool(show)

    @property
    def user_function_name(self) -> str:
        """Returns the name of the user-defined function.

        This is the value that *GetFunctionName* returns. If a user-defined
        function is not used, an empty string is returned.
        See :ref:`nlf-user-defined-function`.
        """
        return self._user_function_name

    def version(self) -> str:
        """Get the version number of the DLL.

        Returns:
        -------
        str
            The version of the DLL.
        """
        if self._version:
            return self._version

        if isinstance(self._dll, ClientNLF):  # noqa: SIM108
            ver = self._dll.dll_version()
        else:
            ver = version(self._dll.lib)  # type: ignore[union-attr]

        self._version = ver
        return ver


class CompositeModel(Model):
    """Combine two models."""

    def __init__(self, op: str, left: Model, right: Model, **kwargs: Any) -> None:  # noqa: ANN401
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
        if op not in ("+", "-", "*", "/"):
            msg = f"Unsupported operator {op!r}"
            raise ValueError(msg)

        rhs = right.equation
        lhs = left.equation

        # remove the scaling factor or the offset from the right equation if it
        # also appears in the left equation for the appropriate binary operator
        index_offset = 0
        lhs_params = _n_params_regex.findall(lhs)
        rhs_params = _n_params_regex.findall(rhs)
        if (op in "*/") and (left._factor in lhs_params and right._factor in rhs_params):  # noqa: SIM114, SLF001
            rhs = right._composite_equation  # noqa: SLF001
            index_offset = 1
        elif (op in "+-") and (left._offset in lhs_params and right._offset in rhs_params):  # noqa: SLF001
            rhs = right._composite_equation  # noqa: SLF001
            index_offset = 1

        # increment the parameter index in the right equation based on
        # how many parameters there are in the left equation
        sub = []
        i, end, n = 0, 0, left.num_parameters
        for match in _n_params_regex.finditer(rhs):
            start, end = match.span()
            new_index = int(match.group()[1:]) + n - index_offset
            sub.append(f"{rhs[i:start]}a{new_index}")
            i = end
        sub.append(rhs[end:])
        rhs = "".join(sub)

        equation = f"({lhs}){op}({rhs})" if rhs else lhs
        super().__init__(equation, **kwargs)


class LoadedModel(Model):
    """A :class:`.Model` that was loaded from a **.nlf** file."""

    def __init__(self, equation: str, *, dll: str | None = None, **options: Any) -> None:  # noqa: ANN401
        """A :class:`.Model` that was loaded from a **.nlf** file.

        Do not instantiate this class directly. The proper way to load a
        **.nlf** file is via the :func:`~msl.nlf.load` function.

        Parameters
        ----------
        equation
            The fit equation. See :class:`.Model` for more details.
        dll
            The path to a non-linear fit DLL file. See :class:`.Model` for
            more details.
        **options
            All additional keyword arguments are passed to :meth:`~.Model.options`.
        """
        super().__init__(equation, dll=dll, **options)

        self.comments: str = ""
        """Comments that were specified."""

        self.nlf_path: str = ""
        """The path to the **.nlf** file that was loaded."""

        self.nlf_version: str = ""
        """The DLL version that created the **.nlf** file."""

        self.params: InputParameters = InputParameters()
        """Input parameters to the fit model."""

        self.ux: NDArray[np.float64] = np.empty(0)
        """Standard uncertainties in the x (stimulus) data."""

        self.uy: NDArray[np.float64] = np.empty(0)
        """Standard uncertainties in the y (response) data."""

        self.x: NDArray[np.float64] = np.empty(0)
        """The independent variable(s) (stimulus) data."""

        self.y: NDArray[np.float64] = np.empty(0)
        """The dependent variable (response) data."""

    def __repr__(self) -> str:
        """Return object representation."""
        # add indentation to the parameters
        if not self.params:
            param_str = "InputParameters()"
        else:
            indent = " " * 4
            params = [indent]
            params.extend(str(self.params).splitlines())
            params[-1] = ")"
            param_str = f"\n{indent}".join(params)

        p = Path(__file__)
        path = p.suffix if p.parent == self._dll_path.parent else self._dll_path

        return (
            f'{self.__class__.__name__}(\n'
            f'  comments={self.comments!r}\n'
            f'  dll={path!r}\n'
            f'  equation={self.equation!r}\n'
            f'  nlf_path={self.nlf_path!r}\n'
            f'  nlf_version={self.nlf_version!r}\n'
            f'  params={param_str}\n'
            f'  ux={np.array2string(self.ux, prefix="     ")}\n'
            f'  uy={np.array2string(self.uy, prefix="     ")}\n'
            f'  x={np.array2string(self.x, prefix="    ")}\n'
            f'  y={np.array2string(self.y, prefix="    ")}\n'
            f')'
        )
