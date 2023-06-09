"""
Call functions in a 32-bit version of the DLL from 64-bit Python.
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
        define_fit_fcn(self.lib, True)

    def dll_version(self) -> str:
        """Get the version number from the DLL."""
        return version(self.lib)

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
                result[k] = np.asarray(v)

        return result
