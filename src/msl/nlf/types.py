"""Custom type annotations."""

from __future__ import annotations

from array import array
from ctypes import Array, _NamedFuncPointer, _Pointer, c_bool, c_double
from typing import Sequence, TypeAlias, TypedDict, TypeVar

from numpy import bool_, dtype, float64
from numpy.ctypeslib import _ndptr
from numpy.typing import NDArray

from .parameters import InputParameter

ArrayLike1D: TypeAlias = Sequence[float] | NDArray[float64]
"""A 1-dimensional, array-like sequence."""

ArrayLike2D: TypeAlias = Sequence[Sequence[float]] | NDArray[float64]
"""A 2-dimensional, array-like sequence."""

EvaluateArray = TypeVar("EvaluateArray", array[float], NDArray[float64])

CtypesOrNumpyDouble: TypeAlias = Array[c_double] | NDArray[float64]
CtypesOrNumpyBool: TypeAlias = Array[c_bool] | NDArray[bool_]

# Delphi array pointers
PMultiData: TypeAlias = type[_Pointer[Array[c_double]]] | type[_ndptr[dtype[float64]]]
PBoolParamData: TypeAlias = type[_Pointer[Array[c_bool]]] | type[_ndptr[dtype[bool_]]]
PSquareMatrix: TypeAlias = type[_Pointer[Array[Array[c_double]]]] | type[_ndptr[dtype[float64]]]
PData = PMultiData
PRealParamData = PMultiData
PIsCorrelated = PBoolParamData

# A user-defined function
GetFunctionValue: TypeAlias = _NamedFuncPointer


class CorrDict(TypedDict):
    """Correlation dictionary."""

    corr: float | NDArray[float64]
    names: tuple[str, str]


class UserDefinedDict(TypedDict):
    """The UserDefined dataclass as a dictionary."""

    equation: str
    function: None
    name: str
    num_parameters: int
    num_variables: int


# Different types that can be iterated over to create a Parameter object
InputParameterType: TypeAlias = (
    InputParameter
    | tuple[str, float]
    | tuple[str, float, bool]
    | tuple[str, float, bool, str | None]
    | list[str | float | bool | None]
    | dict[str, str | float | bool | None]
)
"""Allowed types to create an [InputParameter][msl.nlf.parameters.InputParameter]."""
