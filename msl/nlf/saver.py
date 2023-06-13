"""
Create a **.nlf** file to be read by the Delphi GUI.
"""
import math
from struct import pack


class Saver:

    def __init__(self) -> None:
        """Helper class to create a **.nlf** file."""
        self._buffer = bytearray()

    def save(self, path) -> None:
        """Save the buffer to a file.

        Parameters
        ----------
        path
            The path to save the buffer to.
        """
        with open(path, mode='wb') as fp:
            fp.write(self._buffer)

    def write_boolean(self, value: bool) -> None:
        """Write a boolean."""
        self._buffer.extend(pack('?', value))

    def write_byte(self, value: int) -> None:
        """Write a byte."""
        self._buffer.extend(pack('b', value))

    def write_bytes(self, value: bytes) -> None:
        """Write bytes."""
        self._buffer.extend(pack(f'{len(value)}s', value))

    def write_extended(self, value: float) -> None:
        """Write a Delphi 10-byte extended float."""
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
        """Write an unsigned integer."""
        self._buffer.extend(pack('I', value))

    def write_string(self, value: str) -> None:
        """Write a string."""
        length = len(value)
        self.write_integer(length)
        self._buffer.extend(pack(f'{length}s', value))

    def write_word(self, value: int) -> None:
        """Write an unsigned short."""
        self._buffer.extend(pack('H', value))