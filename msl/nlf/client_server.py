"""
Call functions in a 32-bit DLL from 64-bit Python.
"""
from __future__ import annotations

from array import array
from ctypes import POINTER
from ctypes import c_bool
from ctypes import c_double
from ctypes import cast

from msl.loadlib import Client64
from msl.loadlib import Server32

if Server32.is_interpreter():
    from dll import *
else:
    import numpy as np
    from .dll import UserDefined


class ServerNLF(Server32):

    def __init__(self, host: str, port: int, path: str = '') -> None:
        """Handle requests for the 32-bit DLL.

        Parameters
        ----------
        host
            The IP address of the server.
        port
            The port to run the server on.
        path
            The path to the DLL file.
        """
        super().__init__(path, 'cdll', host, port)
        self._ua = real_param_data()
        self._covar = square_matrix()
        self._user_function = None
        define_fit_fcn(self.lib, True)

    def dll_version(self) -> str:
        """Get the version number from the DLL."""
        return version(self.lib)

    def evaluate(self,
                 a: array,
                 x: array,
                 shape: tuple[int, int]) -> array:
        """Evaluate the user-defined function.

        Parameters
        ----------
        a
            Parameter values.
        x
            Independent variable (stimulus) data.
        shape
            The shape of the *x* data.

        Returns
        -------
        :class:`~array.array`
            Dependent variable (response) data.
        """
        _, npts = shape
        y = array('d', [0.0 for _ in range(npts)])
        return evaluate(self._user_function, a, x, shape, y)

    def fit(self, **kwargs) -> dict:
        """Fit the model to the data using the supplied keyword arguments."""
        kw = {'covar': self._covar, 'ua': self._ua}

        # Create a ctypes memory view to each array to avoid copying values
        for k, v in kwargs.items():
            if isinstance(v, array):
                address, length = v.buffer_info()
                c_type = c_bool if v.itemsize == 1 else c_double
                kw[k] = cast(address, POINTER(c_type * length))
            else:
                kw[k] = v

        # Perform the fit
        result = fit(self.lib, **kw)

        # result['a'] is a ctypes POINTER to the original input array, which
        # shares the same memory space and was updated by the DLL, so return
        # a slice of the original input array. An array.array() object can be
        # pickled but a POINTER cannot be pickled.
        result['a'] = kwargs['a'][:kwargs['nparams']]

        return result

    @staticmethod
    def get_user_defined(directory: str) -> dict:
        """Get all user-defined functions.

        Parameters
        ----------
        directory
            The directory to look for the user-defined functions.

        Returns
        -------
        :class:`dict`
            The user-defined functions.
        """
        new = {}
        for k, v in get_user_defined(directory).items():
            new[k] = v.to_dict()
        return new

    def load_user_defined(self, equation: str, directory: str) -> None:
        """Load a user-defined function in a custom DLL.

        Parameters
        ----------
        equation
            The equation to load.
        directory
            The directory to look for the user-defined function.
        """
        for v in get_user_defined(directory).values():
            if v.equation == equation:
                self._user_function = v.function
                self._user_function.restype = None
                self._user_function.argtypes = [
                    POINTER(c_double), POINTER(c_double), POINTER(c_double)]
                break


class ClientNLF(Client64):

    def __init__(self, path: str) -> None:
        """Send requests to the 32-bit DLL.

        Parameters
        ----------
        path
            The path to the DLL file.
        """
        super().__init__(__file__, path=path)
        self.lib = None  # avoids inspection warnings in PyCharm IDE

    def dll_version(self) -> str:
        """Get the version number from the DLL."""
        return self.request32('dll_version')

    def evaluate(self,
                 a: array,
                 x: array,
                 shape: tuple[int, int]) -> array:
        """Evaluate the user-defined function.

        Parameters
        ----------
        a
            Parameter values.
        x
            Independent variable (stimulus) data.
        shape
            The shape of the *x* data.

        Returns
        -------
        :class:`~array.array`
            Dependent variable (response) data.
        """
        return self.request32('evaluate', a, x, shape)

    def fit(self, **kwargs) -> dict:
        """Fit the model to the data using the supplied keyword arguments."""

        # The 32-bit server does not have 32-bit numpy installed, so convert a
        # numpy ndarray to a builtin array so that the values can be pickled.
        # Using array.array() is faster than using ndarray.flatten().tolist()
        # to get the ndarray to a picklable data type. Also, using a 1D ctypes
        # array on the 32-bit server is easier (and faster) than creating a
        # 2D ctypes array. This is valid since the ndarray's are C_CONTIGUOUS.
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                typecode = 'b' if v.dtype.itemsize == 1 else 'd'
                kwargs[k] = array(typecode, v.tobytes())

        # Send request to the 32-bit server to perform the fit
        result = self.request32('fit', **kwargs)

        # Convert a list back into a numpy ndarray
        for k, v in result.items():
            if isinstance(v, list):
                result[k] = np.array(v)

        return result

    def get_user_defined(self, directory: str) -> dict[str, UserDefined]:
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
        for k, v in self.request32('get_user_defined', directory).items():
            functions[k] = UserDefined(**v)
        return functions

    def load_user_defined(self, equation: str, directory: str) -> None:
        """Load a user-defined function in a custom DLL.

        Parameters
        ----------
        equation
            The equation to load.
        directory
            The directory to look for the user-defined function.
        """
        self.request32('load_user_defined', equation, directory)
