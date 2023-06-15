"""
Save a **.nlf** file.
"""
import math
import os
from struct import pack


class Saver:

    def __init__(self) -> None:
        """Helper class to create a **.nlf** file."""
        self._buffer = bytearray()

    def save(self, path: str, *, overwrite: bool = False) -> None:
        """Save the buffer to a **.nlf** file.

        Parameters
        ----------
        path
            The **.nlf** file path.
        overwrite
            Whether to overwrite the file if it already exists.
        """
        if not overwrite and os.path.isfile(path):
            raise FileExistsError(f'Will not overwrite {path!r}')

        with open(path, mode='wb') as fp:
            fp.write(self._buffer)

    def write_boolean(self, value: bool) -> None:
        """Write a boolean.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        self._buffer.extend(pack('?', value))

    def write_byte(self, value: int) -> None:
        """Write a byte.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        self._buffer.extend(pack('b', value))

    def write_bytes(self, value: bytes) -> None:
        """Write bytes.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        self._buffer.extend(pack(f'{len(value)}s', value))

    def write_extended(self, value: float) -> None:
        """Write a Delphi 10-byte extended float.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        # See Loader.read_extended for structure of an 80-bit float
        if math.isfinite(value):
            mantissa, exponent = math.frexp(abs(value))
            uint16 = exponent + 16382
            if value < 0:
                uint16 |= 0x8000
            uint64 = round(mantissa * (2 << 63))
        elif math.isnan(value):
            uint16, uint64 = 0x7FFF, 1
        else:  # +-Inf
            uint16, uint64 = 0x7FFF, 0
            if value < 0:
                uint16 |= 0x8000
        self._buffer.extend(pack('QH', uint64, uint16))

    def write_integer(self, value: int) -> None:
        """Write an unsigned integer.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        self._buffer.extend(pack('I', value))

    def write_string(self, value: str) -> None:
        """Write a string.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        length = len(value)
        self.write_integer(length)
        self._buffer.extend(pack(f'{length}s', value))

    def write_word(self, value: int) -> None:
        """Write an unsigned short.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        self._buffer.extend(pack('H', value))
