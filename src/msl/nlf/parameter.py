"""Parameters are used as inputs to and results from a fit model."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from .dll import NPAR

if TYPE_CHECKING:
    from typing import Any, Iterable, Iterator

    from numpy.typing import NDArray

    from .types import InputParameterType

_name_regex = re.compile(r"^a(?P<i>\d+)$")


def _check_name(name: str) -> tuple[str, int]:
    """Make sure that the parameter name is valid."""
    match = _name_regex.match(name)
    if not match:
        msg = f"Invalid parameter name {name!r}"
        raise ValueError(msg)
    i = int(match["i"])
    if not (1 <= i <= NPAR):
        msg = f"Invalid parameter name {name!r}, index outside of range"
        raise ValueError(msg)
    return name, i


class Parameter:
    """A generic parameter used as an input to or a result from a fit model."""

    def __init__(self, name: str, value: float, *, label: str | None = None) -> None:
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
        """A custom label associated with the parameter.

        For example, if the fit equation is **a1+a2*x**, you could assign a label of *intercept*
        to *a1* and *slope* to *a2*.
        """
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
        msg = f"Cannot change {self.__class__.__name__} name to {name!r}"
        raise PermissionError(msg)

    @property
    def value(self) -> float:
        """The value of the parameter."""
        raise NotImplementedError


class InputParameter(Parameter):
    """A parameter to use as an input to a fit model."""

    def __init__(self, name: str, value: float, *, constant: bool = False, label: str | None = None) -> None:
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
        """Object representation."""
        return (
            f"InputParameter("
            f"name={self._name!r}, "
            f"value={self._value}, "
            f"constant={self._constant}, "
            f"label={self._label!r})"
        )

    @property
    def constant(self) -> bool:
        """Whether the parameter is held constant (`True`) or allowed to vary (`False`) during the fitting process."""
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
    """A parameter that is returned from a fit model."""

    def __init__(self, name: str, value: float, uncert: float, *, label: str | None = None) -> None:
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
        """Object representation."""
        return (
            f"ResultParameter("
            f"name={self._name!r}, "
            f"value={self._value}, "
            f"uncert={self._uncert}, "
            f"label={self._label!r})"
        )

    @property
    def uncert(self) -> float:
        """The standard uncertainty of the parameter."""
        return self._uncert

    @uncert.setter
    def uncert(self, uncert: float) -> None:
        msg = f"Cannot change ResultParameter uncertainty to {uncert}"
        raise PermissionError(msg)

    @property
    def value(self) -> float:
        """The value of the parameter."""
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        msg = f"Cannot change ResultParameter value to {value}"
        raise PermissionError(msg)


T = TypeVar("T", InputParameter, ResultParameter)
"""Generic parameter type."""


class Parameters(Generic[T]):
    """Base class for a collection of parameters."""

    def __init__(self) -> None:
        """Base class for a collection of parameters."""
        self._map: dict[str, T] = {}

    def __contains__(self, name_or_label: str) -> bool:
        """Check for membership."""
        try:
            self[name_or_label]
        except KeyError:
            return False
        else:
            return True

    def __getitem__(self, name_or_label: str) -> T:
        """Get an item."""
        try:
            return self._map[name_or_label]
        except KeyError:
            pass

        if name_or_label:
            for p in self:
                if p.label == name_or_label:
                    return p

        msg = f"{name_or_label!r} is not a valid 'name' or 'label' of a parameter"
        raise KeyError(msg)

    def __iter__(self) -> Iterator[T]:
        """Return an iterator."""
        return iter(sorted(self._map.values(), key=lambda p: p._i))  # noqa: SLF001

    def __len__(self) -> int:
        """Return the number of parameters."""
        return len(self._map)

    def __repr__(self) -> str:
        """Return the object representation."""
        if not self._map:
            return f"{self.__class__.__name__}()"

        params = ",\n  ".join(f"{p}" for p in self)
        return f"{self.__class__.__name__}(\n  {params}\n)"

    def labels(self) -> list[str | None]:
        """Returns the :attr:`~msl.nlf.parameter.Parameter.label` of each parameter."""
        return [p.label for p in self]

    def names(self) -> list[str]:
        """Returns the :attr:`~msl.nlf.parameter.Parameter.name` of each parameter."""
        return [p.name for p in self]

    def values(self) -> NDArray[np.float64]:
        """Returns the :attr:`~msl.nlf.parameter.Parameter.value` of each parameter."""
        return np.array([p.value for p in self], dtype=float)


class InputParameters(Parameters[InputParameter]):
    r"""A collection of :class:`.InputParameter`\\s for a fit model."""

    def __init__(self, parameters: Iterable[InputParameterType] | None = None) -> None:
        r"""A collection of :class:`.InputParameter`\\s for a fit model.

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
        """Delete an item."""
        p = self[name_or_label]
        del self._map[p.name]

    def __setitem__(self, name: str, obj: Any) -> None:  # noqa: ANN401
        """Set an item."""
        if isinstance(obj, InputParameter):
            parameter = obj
        elif isinstance(obj, (int, float)):
            parameter = self._create_parameter(name, obj)
        elif isinstance(obj, dict):
            obj.setdefault("name", name)
            parameter = self._create_parameter(**obj)
        else:
            first, *rest = obj
            if not isinstance(first, str):
                obj = (name, first, *rest)
            parameter = self._create_parameter(*obj)

        if name != parameter.name:
            msg = f"name != parameter.name [{name!r} != {parameter.name!r}]"
            raise ValueError(msg)

        if parameter.name in self._map:
            msg = f"A parameter named {parameter.name!r} has already been added"
            raise ValueError(msg)

        self._map[name] = parameter

    @staticmethod
    def _create_parameter(*args: Any, **kwargs: Any) -> InputParameter:  # noqa: ANN401
        if kwargs:
            for k in ("name", "value"):
                if k not in kwargs:
                    msg = f"Must specify the {k!r} of the InputParameter"
                    raise ValueError(msg)
            return InputParameter(**kwargs)

        if len(args) == 1:
            if not isinstance(args[0], InputParameter):
                msg = "Must be an InputParameter object " "if specifying only one argument"
                raise TypeError(msg)
            return args[0]

        constant, label = False, None
        if len(args) == 2:  # noqa: PLR2004
            name, value = args
        elif len(args) == 3:  # noqa: PLR2004
            name, value, constant = args
        elif len(args) == 4:  # noqa: PLR2004
            name, value, constant, label = args
        else:
            msg = "Too many arguments specified"
            raise ValueError(msg)

        return InputParameter(name, value, constant=constant, label=label)

    def add(self, *args: Any, **kwargs: Any) -> InputParameter:  # noqa: ANN401
        """Add an :class:`.InputParameter`.

        An :class:`.InputParameter` can be added using either positional or
        keyword arguments, but you cannot use both simultaneously. You can
        specify positional arguments by using one of four options:

            * InputParameter (a single argument must be an :class:`.InputParameter`)
            * name, value
            * name, value, constant
            * name, value, constant, label

        You could alternatively add an :class:`.InputParameter` in the same way
        that you add items to a :class:`dict`

        Returns:
        -------
        :class:`.InputParameter`
            The input parameter that was added.

        Examples:
        --------
            >>> from msl.nlf import InputParameter, InputParameters
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
            msg = "Cannot specify both positional and keyword arguments"
            raise ValueError(msg)

        if not (args or kwargs):
            msg = "Must specify either positional or keyword arguments"
            raise ValueError(msg)

        p = self._create_parameter(*args, **kwargs)
        self[p.name] = p
        return p

    def add_many(self, parameters: Iterable[InputParameterType]) -> None:
        r"""Add many :class:`.InputParameter`\\s.

        Parameters
        ----------
        parameters
            An iterable of either :class:`.InputParameter` instances or objects
            that can be used to create an :class:`.InputParameter` instance.
            See :meth:`.add` for more examples.

        Examples:
        --------
            >>> from msl.nlf import InputParameter, InputParameters
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
                msg = f"Cannot create an InputParameter from only a string, {p!r}"
                raise TypeError(msg)
            else:
                self.add(*p)

    def clear(self) -> None:
        r"""Remove all :class:`.InputParameter`\\s from the collection."""
        self._map.clear()

    def constants(self) -> NDArray[np.bool_]:
        """Returns the :attr:`~msl.nlf.parameter.InputParameter.constant` of each parameter."""
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

        Returns:
        -------
        InputParameter
            The input parameter that was popped.
        """
        p = self[name_or_label]
        return self._map.pop(p.name)

    def update(self, name_or_label: str, **attribs: Any) -> None:  # noqa: ANN401
        """Update the attributes of an :class:`.InputParameter`.

        Parameters
        ----------
        name_or_label
            The :attr:`~msl.nlf.parameter.Parameter.name` or
            :attr:`~msl.nlf.parameter.Parameter.label` of an
            :class:`.InputParameter`.
        **attribs
            The new attributes.

        Examples:
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
            if k == "value":
                parameter.value = v
            elif k == "constant":
                parameter.constant = v
            elif k == "label":
                parameter.label = v
            elif k == "name" and name_or_label != v:
                parameter.name = v


class ResultParameters(Parameters[ResultParameter]):
    r"""A collection of :class:`.ResultParameter`\\s from a fit model."""

    def __init__(self, result: dict[str, Any], params: InputParameters) -> None:
        r"""A collection of :class:`.ResultParameter`\\s from a fit model.

        Parameters
        ----------
        result
            The result from a fit model.
        params
            The input parameters to the fit model.
        """
        super().__init__()
        a, ua = result.pop("a"), result.pop("ua")
        names, labels = params.names(), params.labels()
        for name, value, uncert, label in zip(names, a, ua, labels):
            self._map[name] = ResultParameter(name, value, uncert, label=label)

    def __delitem__(self, name_or_label: str) -> None:
        """Delete an item."""
        msg = "Cannot delete a ResultParameter"
        raise PermissionError(msg)

    def __setitem__(self, name: str, obj: Any) -> None:  # noqa: ANN401
        """Set an item."""
        msg = "Cannot set a ResultParameter"
        raise PermissionError(msg)

    def uncerts(self) -> NDArray[np.float64]:
        """Returns the :attr:`~msl.nlf.parameter.ResultParameter.uncert` of each parameter."""
        return np.array([p.uncert for p in self], dtype=np.float64)
