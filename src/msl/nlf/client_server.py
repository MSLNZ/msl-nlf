"""Call user-defined functions in a 32-bit DLL from 64-bit Python."""

from __future__ import annotations

from array import array
from ctypes import POINTER, c_bool, c_double, cast
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from msl.loadlib import Client64, Server32  # type: ignore[import-untyped]
except ModuleNotFoundError:
    Client64 = object

    class Server32:  # type: ignore[no-redef]
        """Mocked Server."""

        @staticmethod
        def is_interpreter() -> bool:
            """Mocked is_interpreter method."""
            return False


if Server32.is_interpreter():
    from delphi import (  # type: ignore[import-not-found]
        define_fit_fcn,
        delphi_version,
        evaluate,
        fit,
        get_user_defined,
        real_param_data,
        square_matrix,
    )
else:
    import numpy as np

    from .delphi import UserDefined

if TYPE_CHECKING:
    from ctypes import CDLL
    from typing import Any

    from .types import GetFunctionValue, UserDefinedDict


class ServerNLF(Server32):  # type: ignore[misc]
    """Handle requests for the 32-bit Delphi library."""

    def __init__(self, host: str, port: int, path: str = "") -> None:
        """Handle requests for the 32-bit Delphi library.

        Args:
            host: The IP address of the server.
            port: The port to run the server on.
            path: The path to the Delphi shared-library file.
        """
        super().__init__(path, "cdll", host, port)
        self._user_function: GetFunctionValue
        self._ua = real_param_data()
        self._covar = square_matrix()
        define_fit_fcn(self.lib, as_ctypes=True)

    def delphi_version(self) -> str:
        """Get the version number of the Delphi shared library.

        Returns:
            The version number.
        """
        v: str = delphi_version(self.lib)
        return v

    def evaluate(self, a: array[float], x: array[float], shape: tuple[int, int]) -> array[float]:
        """Evaluate the user-defined function.

        Args:
            a: Parameter values.
            x: Independent variable (stimulus) data.
            shape: The shape of the *x* data.

        Returns:
            Dependent variable (response) data.
        """
        _, npts = shape
        y = array("d", [0.0 for _ in range(npts)])
        result: array[float] = evaluate(self._user_function, a, x, shape, y)
        return result

    def fit(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Fit the model to the data using the supplied keyword arguments.

        Returns:
            The fit result.
        """
        kw = {"covar": self._covar, "ua": self._ua}

        # Create a ctypes memory view to each array to avoid copying values
        for k, v in kwargs.items():
            if isinstance(v, array):
                address, length = v.buffer_info()
                c_type = c_bool if v.itemsize == 1 else c_double
                kw[k] = cast(address, POINTER(c_type * length))  # type: ignore[arg-type, operator]
            else:
                kw[k] = v

        # Perform the fit
        result: dict[str, Any] = fit(lib=self.lib, **kw)

        # result["a"] is a ctypes POINTER to the original input array, which
        # shares the same memory space and was updated by the DLL, so return
        # a slice of the original input array. An array.array() object can be
        # pickled but a POINTER cannot be pickled.
        result["a"] = kwargs["a"][: kwargs["nparams"]]

        return result

    @staticmethod
    def get_user_defined(directory: str, extension: str) -> dict[str, UserDefinedDict]:
        """Get all user-defined functions.

        Args:
            directory: The directory to look for the user-defined functions.
            extension: The file extension for the user-defined functions.

        Returns:
            The user-defined functions.
        """
        return {k: v.to_dict() for k, v in get_user_defined(directory, extension).items()}

    def load_user_defined(self, equation: str, directory: str, extension: str) -> None:
        """Load a user-defined function in a custom DLL.

        Args:
            equation: The equation to load.
            directory: The directory to look for the user-defined function.
            extension: The file extension for the user-defined functions.
        """
        for v in get_user_defined(directory, extension).values():
            if v.equation == equation:
                self._user_function = v.function
                self._user_function.restype = None
                self._user_function.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double)]
                break


class ClientNLF(Client64):  # type: ignore[misc]
    """Send requests to the 32-bit Delphi library."""

    def __init__(self, path: str | Path) -> None:
        """Send requests to the 32-bit Delphi library.

        Args:
            path: The path to the Delphi library.
        """
        super().__init__(__file__, path=str(path))
        self.lib: CDLL

    def delphi_version(self) -> str:
        """Get the version number of the Delphi shared library.

        Returns:
            The version number.
        """
        response: str = self.request32("delphi_version")
        return response

    def evaluate(self, a: array[float], x: array[float], shape: tuple[int, int]) -> array[float]:
        """Evaluate the user-defined function.

        Args:
            a: Parameter values.
            x: Independent variable (stimulus) data.
            shape: The shape of the *x* data.

        Returns:
            Dependent variable (response) data.
        """
        response: array[float] = self.request32("evaluate", a, x, shape)
        return response

    def fit(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Fit the model to the data using the supplied keyword arguments.

        Returns:
            The fit results.
        """
        # The 32-bit server does not have 32-bit numpy installed, so convert a
        # numpy ndarray to a builtin array so that the values can be pickled.
        # Using array.array() is faster than using ndarray.flatten().tolist()
        # to get the ndarray to a picklable data type. Also, using a 1D ctypes
        # array on the 32-bit server is easier (and faster) than creating a
        # 2D ctypes array. This is valid since the ndarray's are C_CONTIGUOUS.
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                type_code = "b" if v.dtype.itemsize == 1 else "d"
                kwargs[k] = array(type_code, v.tobytes())

        # Send request to the 32-bit server to perform the fit
        result: dict[str, Any] = self.request32("fit", **kwargs)

        # Convert a list back into a numpy ndarray
        for k, v in result.items():
            if isinstance(v, list):
                result[k] = np.array(v)

        return result

    def get_user_defined(self, directory: str | Path, extension: str) -> dict[Path, UserDefined]:
        """Get all user-defined functions.

        Args:
            directory: The directory to look for the user-defined functions.
            extension: The file extension for the user-defined functions.

        Returns:
            The user-defined functions.
        """
        response = self.request32("get_user_defined", str(directory), extension)
        return {Path(k): UserDefined(**v) for k, v in response.items()}

    def load_user_defined(self, equation: str, directory: str | Path, extension: str) -> None:
        """Load a user-defined function.

        Args:
            equation: The equation to load.
            directory: The directory to look for the user-defined function.
            extension: The file extension for the user-defined functions.
        """
        self.request32("load_user_defined", equation, str(directory), extension)
