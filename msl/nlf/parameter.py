"""
Parameters are used as inputs to and results from a fit model.
"""
from __future__ import annotations

import re
from typing import Any
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np

from .dll import NPAR

_name_regex = re.compile(r'^a(?P<i>\d+)$')


def _check_name(name: str) -> Tuple[str, int]:
    """Make sure that the parameter name is valid."""
    match = _name_regex.match(name)
    if not match:
        raise ValueError(f'Invalid parameter name {name!r}')
    i = int(match['i'])
    if not (1 <= i <= NPAR):
        raise ValueError(f'Invalid parameter name {name!r}, '
                         f'index outside of range')
    return name, i


class Parameter:

    def __init__(self,
                 name: str,
                 value: float,
                 *,
                 label: str = None) -> None:
        """A generic parameter used as an input to or a result from a fit model.

        Parameters
        ----------
        name
            The name of the parameter in the equation (e.g., *a1*).
        value
            The value of the parameter.
        label
            A custom label associated with the parameter. For example, if the
            fit equation is **a1+a2*x**, you could assign a label of *intercept*
            to *a1* and *slope* to *a2*.
        """
        name, i = _check_name(name)
        self._name: str = name
        self._value: float = float(value)
        self._label: str | None = label

        # keeping a record of the `i` value (in the `a_i` name)
        # is required when sorting a collection of parameters
        # i.e., 'a10' < 'a2' == True, whereas 10 < 2 == False
        self._i: int = i

    @property
    def label(self) -> str | None:
        """A custom label associated with the parameter. For example, if the
        fit equation is **a1+a2*x**, you could assign a label of *intercept*
        to *a1* and *slope* to *a2*."""
        return self._label

    @label.setter
    def label(self, label: str | None) -> None:
        self._label = label

    @property
    def name(self) -> str:
        """The name of the parameter in the equation (e.g., *a1*)."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        raise PermissionError(f'Cannot change {self.__class__.__name__} name')

    @property
    def value(self) -> float:
        """The value of the parameter."""
        raise NotImplementedError


class InputParameter(Parameter):

    def __init__(self,
                 name: str,
                 value: float,
                 *,
                 constant: bool = False,
                 label: str = None) -> None:
        """A parameter to use as an input to a fit model.

        Parameters
        ----------
        name
            The name of the parameter in the equation (e.g., *a1*).
        value
            The value of the parameter.
        constant
            Whether the parameter is held constant (:data:`True`) or allowed
            to vary (:data:`False`) during the fitting process.
        label
            A custom label associated with the parameter. For example, if the
            fit equation is **a1+a2*x**, you could assign a label of *intercept*
            to *a1* and *slope* to *a2*.
        """
        super().__init__(name, value, label=label)
        self._constant: bool = bool(constant)

    def __repr__(self) -> str:
        return f'InputParameter(' \
               f'name={self._name!r}, ' \
               f'value={self._value}, ' \
               f'constant={self._constant}, ' \
               f'label={self._label!r})'

    @property
    def constant(self) -> bool:
        """Whether the parameter is held constant (:data:`True`) or
        allowed to vary (:data:`False`) during the fitting process."""
        return self._constant

    @constant.setter
    def constant(self, constant: bool) -> None:
        self._constant = bool(constant)

    @property
    def value(self) -> float:
        """The value of the parameter."""
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = float(value)


class ResultParameter(Parameter):

    def __init__(self,
                 name: str,
                 value: float,
                 uncert: float,
                 *,
                 label: str = None) -> None:
        """A parameter that is returned from a fit model.

        Parameters
        ----------
        name
            The name of the parameter in the equation (e.g., *a1*).
        value
            The value of the parameter.
        uncert
            The standard uncertainty of the parameter.
        label
            A custom label associated with the parameter. For example, if the
            fit equation is **a1+a2*x**, you could assign a label of *intercept*
            to *a1* and *slope* to *a2*.
        """
        super().__init__(name, value, label=label)
        self._uncert: float = float(uncert)

    def __repr__(self) -> str:
        return f'ResultParameter(' \
               f'name={self._name!r}, ' \
               f'value={self._value}, ' \
               f'uncert={self._uncert}, ' \
               f'label={self._label!r})'

    @property
    def uncert(self) -> float:
        """The standard uncertainty of the parameter."""
        return self._uncert

    @uncert.setter
    def uncert(self, uncert: float) -> None:
        raise PermissionError('Cannot change ResultParameter uncertainty')

    @property
    def value(self) -> float:
        """The value of the parameter."""
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        raise PermissionError('Cannot change ResultParameter value')


# Different types that can be iterated over to create a Parameter object
InputParameterType = Union[
    InputParameter,
    Tuple[str, float],
    Tuple[str, float, bool],
    Tuple[str, float, bool, Union[str, None]],
    List,
    Dict[str, Union[str, float, bool, None]]
]
"""Allowed types to create an :class:`~msl.nlf.parameter.InputParameter`."""

T = TypeVar('T', InputParameter, ResultParameter)
"""Generic parameter type."""


class Parameters(Generic[T]):

    def __init__(self) -> None:
        """Base class for a collection of parameters."""
        self._map: dict[str, T] = {}

    def __contains__(self, name_or_label: str) -> bool:
        try:
            self[name_or_label]
        except KeyError:
            return False
        else:
            return True

    def __getitem__(self, name_or_label: str) -> T:
        try:
            return self._map[name_or_label]
        except KeyError:
            pass
        if name_or_label:
            for p in self:
                if p.label == name_or_label:
                    return p
        raise KeyError(f"{name_or_label!r} is not a valid "
                       f"'name' or 'label' of a parameter")

    def __iter__(self) -> Iterator[T]:
        return iter(sorted(self._map.values(), key=lambda p: p._i))  # noqa

    def __len__(self) -> int:
        return len(self._map)

    def __repr__(self) -> str:
        if not self._map:
            return f'{self.__class__.__name__}()'

        params = ',\n  '.join(f'{p}' for p in self)
        return f'{self.__class__.__name__}(\n  {params}\n)'

    def labels(self) -> List[Union[str, None]]:
        """Returns the :attr:`~msl.nlf.parameter.Parameter.label`
        of each parameter."""
        return [p.label for p in self]

    def names(self) -> List[str]:
        """Returns the :attr:`~msl.nlf.parameter.Parameter.name`
        of each parameter."""
        return [p.name for p in self]

    def values(self) -> np.ndarray[float]:
        """Returns the :attr:`~msl.nlf.parameter.Parameter.value`
        of each parameter."""
        return np.array([p.value for p in self], dtype=float)


class InputParameters(Parameters[InputParameter]):

    def __init__(self, parameters: Iterable[InputParameterType] = None) -> None:
        """A collection of :class:`.InputParameter`\\s for a fit model.

        Parameters
        ----------
        parameters
            An iterable of either :class:`.InputParameter` instances or objects
            that can be used to create an :class:`.InputParameter` instance.
            See :meth:`.add` for examples.
        """
        super().__init__()
        if parameters:
            self.add_many(parameters)

    def __delitem__(self, name_or_label: str) -> None:
        p = self[name_or_label]
        del self._map[p.name]

    def __setitem__(self, name: str, obj: Any) -> None:
        if isinstance(obj, InputParameter):
            parameter = obj
        elif isinstance(obj, (int, float)):
            parameter = self._create_parameter(name, obj)
        elif isinstance(obj, dict):
            obj.setdefault('name', name)
            parameter = self._create_parameter(**obj)
        else:
            first, *rest = obj
            if not isinstance(first, str):
                obj = (name, first, *rest)
            parameter = self._create_parameter(*obj)

        if name != parameter.name:
            raise ValueError(f'name != parameter.name '
                             f'[{name!r} != {parameter.name!r}]')
        if parameter.name in self._map:
            raise ValueError(f'A parameter named {parameter.name!r} '
                             f'has already been added')

        self._map[name] = parameter

    @staticmethod
    def _create_parameter(*args, **kwargs) -> InputParameter:
        if kwargs:
            for k in ('name', 'value'):
                if k not in kwargs:
                    raise ValueError(
                        f'Must specify the {k!r} of the InputParameter')
            return InputParameter(**kwargs)

        if len(args) == 1:
            if not isinstance(args[0], InputParameter):
                raise TypeError('Must be an InputParameter object '
                                'if specifying only one argument')
            return args[0]

        constant, label = False, None
        if len(args) == 2:
            name, value = args
        elif len(args) == 3:
            name, value, constant = args
        elif len(args) == 4:
            name, value, constant, label = args
        else:
            raise ValueError('Too many arguments specified')

        return InputParameter(name, value, constant=constant, label=label)

    def add(self, *args, **kwargs) -> InputParameter:
        """Add an :class:`.InputParameter`.

        An :class:`.InputParameter` can be added using either positional or
        keyword arguments, but you cannot mix not both. You can specify
        positional arguments by using one of four options:

            * InputParameter (a single argument must be an :class:`.InputParameter`)
            * name, value
            * name, value, constant
            * name, value, constant, label

        You could alternatively add an :class:`.InputParameter` in the same way
        that you add items to a :class:`dict`

        Returns
        -------
        :class:`.InputParameter`
            The input parameter that was added.

        Examples
        --------

            >>> params = InputParameters()
            >>> a1 = params.add('a1', 1)
            >>> a2 = params.add('a2', 0.34, True)
            >>> a3 = params.add('a3', -1e3, False, 'offset')
            >>> a4 = params.add(name='a4', value=3.14159, constant=True, label='pi')
            >>> a5 = params.add(name='a5', value=100)
            >>> a6 = params.add(InputParameter('a6', 32.0))
            >>> params['a7'] = InputParameter('a7', 7, constant=True)
            >>> params['a8'] = 88.8
            >>> params['a9'] = (-1, True)
            >>> params['a10'] = {'value': 0, 'label': 'intercept'}
            >>> for param in params:
            ...     print(f'{param.name}={param.value}')
            a1=1.0
            a2=0.34
            a3=-1000.0
            a4=3.14159
            a5=100.0
            a6=32.0
            a7=7.0
            a8=88.8
            a9=-1.0
            a10=0.0

        """
        if args and kwargs:
            raise ValueError('Cannot specify both positional and keyword arguments')
        if not (args or kwargs):
            raise ValueError('Must specify either positional or keyword arguments')
        p = self._create_parameter(*args, **kwargs)
        self[p.name] = p
        return p

    def add_many(self, parameters: Iterable[InputParameterType]) -> None:
        """Add many :class:`.InputParameter`\\s.

        Parameters
        ----------
        parameters
            An iterable of either :class:`.InputParameter` instances or objects
            that can be used to create an :class:`.InputParameter` instance.
            See :meth:`.add` for more examples.

        Examples
        --------

            >>> inputs = (InputParameter('a1', 1),
            ...           ('a2', 2, True),
            ...           {'name': 'a3', 'value': 3})
            >>> params = InputParameters()
            >>> params.add_many(inputs)
            >>> for param in params:
            ...     print(param)
            InputParameter(name='a1', value=1.0, constant=False, label=None)
            InputParameter(name='a2', value=2.0, constant=True, label=None)
            InputParameter(name='a3', value=3.0, constant=False, label=None)

        """
        for p in parameters:
            if isinstance(p, InputParameter):
                self.add(p)
            elif isinstance(p, dict):
                self.add(**p)
            elif isinstance(p, str):
                raise TypeError(f'Cannot create an InputParameter '
                                f'from only a string, {p!r}')
            else:
                self.add(*p)

    def clear(self) -> None:
        """Remove all :class:`.InputParameter`\\s from the collection."""
        self._map.clear()

    def constants(self) -> np.ndarray[bool]:
        """Returns the :attr:`~msl.nlf.parameter.InputParameter.constant`
        of each parameter."""
        return np.array([p.constant for p in self], dtype=bool)

    def pop(self, name_or_label: str) -> InputParameter:
        """Pop an :class:`.InputParameter` from the collection.

        This will remove the :class:`.InputParameter` from the collection
        and return it.

        Parameters
        ----------
        name_or_label
            The :attr:`~msl.nlf.parameter.Parameter.name` or
            :attr:`~msl.nlf.parameter.Parameter.label` of an
            :class:`.InputParameter`.

        Returns
        -------
        InputParameter
            The input parameter that was popped.
        """
        p = self[name_or_label]
        return self._map.pop(p.name)

    def update(self, name_or_label: str, **attribs) -> None:
        """Update the attributes of an :class:`.InputParameter`.

        Parameters
        ----------
        name_or_label
            The :attr:`~msl.nlf.parameter.Parameter.name` or
            :attr:`~msl.nlf.parameter.Parameter.label` of an
            :class:`.InputParameter`.
        **attribs
            The new attributes.

        Examples
        --------
        First, add a parameter

            >>> from msl.nlf import InputParameters
            >>> params = InputParameters()
            >>> a1 = params.add('a1', 1)
            >>> a1
            InputParameter(name='a1', value=1.0, constant=False, label=None)

        then update it by calling the :meth:`.update` method

            >>> params.update('a1', value=0, constant=True, label='intercept')
            >>> a1
            InputParameter(name='a1', value=0.0, constant=True, label='intercept')

        Alternatively, you can update a parameter by directly modifying an attribute

            >>> a1.label = 'something-new'
            >>> a1.constant = False
            >>> a1.value = -3.2
            >>> params['a1']
            InputParameter(name='a1', value=-3.2, constant=False, label='something-new')

        """
        parameter = self[name_or_label]
        for k, v in attribs.items():
            if k == 'value':
                parameter.value = v
            elif k == 'constant':
                parameter.constant = v
            elif k == 'label':
                parameter.label = v
            elif k == 'name' and name_or_label != v:
                parameter.name = v


class ResultParameters(Parameters[ResultParameter]):

    def __init__(self, result: dict, params: InputParameters) -> None:
        """A collection of :class:`.ResultParameter`\\s from a fit model.

        Parameters
        ----------
        result
            The result from a fit model.
        params
            The input parameters to the fit model.
        """
        super().__init__()
        a, ua = result.pop('a'), result.pop('ua')
        names, labels = params.names(), params.labels()
        for name, value, uncert, label in zip(names, a, ua, labels):
            self._map[name] = ResultParameter(name, value, uncert, label=label)

    def __delitem__(self, name_or_label: str) -> None:
        raise PermissionError('Cannot delete a ResultParameter')

    def __setitem__(self, name: str, obj: Any) -> None:
        raise PermissionError('Cannot set a ResultParameter')

    def uncerts(self) -> np.ndarray[float]:
        """Returns the :attr:`~msl.nlf.parameter.ResultParameter.uncert`
        of each parameter."""
        return np.array([p.uncert for p in self], dtype=float)
