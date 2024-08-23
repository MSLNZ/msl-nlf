from __future__ import annotations

import numpy as np
import pytest

from msl.nlf import InputParameter, InputParameters
from msl.nlf.dll import NPAR
from msl.nlf.parameter import ResultParameter, ResultParameters


@pytest.mark.parametrize("name", ["a", " a ", "ax", "a 1", "a0", f"a{NPAR+1}"])
def test_invalid_name(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid parameter name"):
        InputParameter(name, 1)
    with pytest.raises(ValueError, match="Invalid parameter name"):
        ResultParameter(name, 1, 0)

    # TypeError comes from regex.match()
    with pytest.raises(TypeError, match="bytes-like object"):
        InputParameter(name.encode(), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes-like object"):
        ResultParameter(name.encode(), 1, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("args", "error"),
    [((), TypeError), ((1,), TypeError), ((1, "a1"), TypeError), (("a1", 1j), TypeError), (("a1", "a2"), ValueError)],
)
def test_invalid_arg_types(args: tuple[int | str | complex], error: type[Exception]) -> None:
    with pytest.raises(error):
        InputParameter(*args)  # type: ignore[call-arg, arg-type]
    with pytest.raises(error):
        ResultParameter(*args, uncert=0)  # type: ignore[call-arg, arg-type]


def test_input_parameter() -> None:
    p = InputParameter("a1", 1)
    assert str(p) == "InputParameter(name='a1', value=1.0, constant=False, label=None)"
    assert p.name == "a1"
    assert p.value == 1.0
    assert isinstance(p.value, float)
    assert p.constant is False
    assert p.label is None

    with pytest.raises(PermissionError, match="Cannot change"):
        p.name = "a2"

    p = InputParameter("a99", 1, constant=True)
    assert str(p) == "InputParameter(name='a99', value=1.0, constant=True, label=None)"
    assert p.name == "a99"
    assert p.value == 1.0
    assert isinstance(p.value, float)
    assert p.constant is True
    assert p.label is None

    # the value will be converted to float
    p.value = -2

    # the value will be converted to bool
    p.constant = 0  # type: ignore[assignment]

    p.label = "slope"
    assert str(p) == "InputParameter(name='a99', value=-2.0, constant=False, label='slope')"

    p = InputParameter("a1", 1, label="label")
    assert str(p) == "InputParameter(name='a1', value=1.0, constant=False, label='label')"
    assert p.name == "a1"
    assert p.value == 1.0
    assert isinstance(p.value, float)
    assert p.constant is False
    assert p.label == "label"

    p = InputParameter("a1", 1, constant=1, label="label")  # type: ignore[arg-type]
    assert str(p) == "InputParameter(name='a1', value=1.0, constant=True, label='label')"
    assert p.name == "a1"
    assert p.value == 1.0
    assert isinstance(p.value, float)
    assert p.constant is True
    assert p.label == "label"


def test_result_parameter() -> None:
    p = ResultParameter("a1", 1, 0)
    assert str(p) == "ResultParameter(name='a1', value=1.0, uncert=0.0, label=None)"
    assert p.name == "a1"
    assert p.value == 1.0
    assert isinstance(p.value, float)
    assert p.uncert == 0.0
    assert isinstance(p.uncert, float)
    assert p.label is None

    with pytest.raises(PermissionError, match="Cannot change"):
        p.name = "a2"
    with pytest.raises(PermissionError, match="Cannot change"):
        p.value = 2.0
    with pytest.raises(PermissionError, match="Cannot change"):
        p.uncert = 1.0

    # only the label is allowed to change
    p.label = "new"
    assert str(p) == "ResultParameter(name='a1', value=1.0, uncert=0.0, label='new')"


def test_add_args_and_kwargs() -> None:
    p = InputParameters()
    with pytest.raises(ValueError, match="Cannot specify both"):
        p.add("a1", 1, constant=True)
    with pytest.raises(ValueError, match="Must specify either"):
        p.add()


def test_add_method() -> None:  # noqa: PLR0915
    p = InputParameters()

    assert len(p) == 0
    assert not p
    assert len(p.names()) == 0

    out = p.add("a1", 1)
    assert out.name == "a1"
    assert out.value == 1.0
    assert out.constant is False
    assert out.label is None
    assert len(p) == 1

    with pytest.raises(ValueError, match="already been added"):
        p.add(name="a1", value=1)

    with pytest.raises(ValueError, match="Must specify the 'name'"):
        p.add(value=1)

    with pytest.raises(ValueError, match="Must specify the 'value'"):
        p.add(name="a2")

    with pytest.raises(ValueError, match="Too many arguments"):
        p.add("a2", 2, True, "label", 99)  # noqa: FBT003

    with pytest.raises(TypeError, match="Must be an InputParameter object"):
        p.add("a1")

    out = p.add("a6", 6, True, "label6")  # noqa: FBT003
    assert out.name == "a6"
    assert out.value == 6.0
    assert out.constant is True
    assert out.label == "label6"
    assert len(p) == 2

    out = p.add(name="a2", value=2, label="label2")
    assert out.name == "a2"
    assert out.value == 2.0
    assert out.constant is False
    assert out.label == "label2"
    assert len(p) == 3

    out = p.add(name="a3", value=3)
    assert out.name == "a3"
    assert out.value == 3.0
    assert out.constant is False
    assert out.label is None
    assert len(p) == 4

    out = p.add("a5", 5, True)  # noqa: FBT003
    assert out.name == "a5"
    assert out.value == 5.0
    assert out.constant is True
    assert out.label is None
    assert len(p) == 5

    with pytest.raises(ValueError, match="already been added"):
        p.add("a3", 1)

    out = p.add(name="a4", value=4, label="label4", constant=True)
    assert out.name == "a4"
    assert out.value == 4.0
    assert out.constant is True
    assert out.label == "label4"
    assert len(p) == 6

    with pytest.raises(TypeError):
        p.add(name="a7", value=5, label="label", constant=True, unknown=4)

    # add more to get past 10
    p.add(name="a10", value=10, constant=True, label="label10")
    p.add(name="a7", value=7, constant=True)
    p.add(name="a12", value=12, constant=True, label="label12")
    p.add(name="a9", value=9, constant=True)
    p.add(name="a11", value=11, constant=True)
    p.add(name="a8", value=8, constant=True, label="label8")

    # must be sorted by name, not in the order each was added
    assert p.names() == ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12"]
    assert np.array_equal(p.values(), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
    assert p.labels() == [
        None,
        "label2",
        None,
        "label4",
        None,
        "label6",
        None,
        "label8",
        None,
        "label10",
        None,
        "label12",
    ]
    assert np.array_equal(
        p.constants(), np.array([False, False, False, True, True, True, True, True, True, True, True, True])
    )

    for item in (p, list(p)):
        for i, param in enumerate(item, start=1):
            assert param.name == f"a{i}"
            assert param.value == i
            if i < 4:
                assert param.constant is False
            else:
                assert param.constant is True
            if i % 2:
                assert param.label is None
            else:
                assert param.label == f"label{i}"

    p.clear()
    assert len(p) == 0


def test_add_variations() -> None:
    p = InputParameters()
    p.add(InputParameter("a1", 1))
    p.add("a2", 2)
    p.add("a3", 3, True)  # noqa: FBT003
    p.add("a4", 4, False, "label")  # noqa: FBT003
    p.add(name="a5", value=5)
    p.add(name="a6", value=6, constant=True)
    p.add(name="a7", value=7, constant=True, label="hi")
    p["a8"] = InputParameter("a8", 1)
    p["a9"] = 9
    p["a10"] = "a10", 10
    p["a11"] = (11, False)
    p["a12"] = "a12", 12, False
    p["a13"] = 13, False, "amp"
    p["a14"] = ("a14", 14, True, "a")
    p["a15"] = {"value": 15}
    p["a16"] = {"name": "a16", "value": 16}
    p["a17"] = {"value": 17, "constant": True}
    p["a18"] = {"value": 18, "constant": True, "label": "18"}

    with pytest.raises(ValueError, match="Invalid parameter name 'a'"):
        # a string is iterable and therefore the value is unpacked to: 'a', ['1', '9']
        p["a19"] = "a19"

    with pytest.raises(ValueError, match="already been added"):
        p["a1"] = InputParameter("a1", 1)

    with pytest.raises(ValueError, match="already been added"):
        p["a1"] = 1

    with pytest.raises(ValueError, match="already been added"):
        p["a1"] = {"name": "a1", "value": 99}

    with pytest.raises(ValueError, match="already been added"):
        p["a1"] = ("a1", 99)

    with pytest.raises(ValueError, match="name != parameter.name"):
        p["a19"] = InputParameter("a20", 1)

    with pytest.raises(ValueError, match="name != parameter.name"):
        p["a19"] = ("a20", 1)

    with pytest.raises(ValueError, match="name != parameter.name"):
        p["a19"] = ("a20", 1, True)

    with pytest.raises(ValueError, match="name != parameter.name"):
        p["a19"] = ("a20", 1, True, "hi")

    with pytest.raises(ValueError, match="name != parameter.name"):
        p["a19"] = {"name": "a20", "value": 1}

    with pytest.raises(TypeError, match="Must be an InputParameter object"):
        p.add({"name": "a19", "value": 5})  # not using **kwargs

    with pytest.raises(TypeError, match="Must be an InputParameter object"):
        p.add(("a19", 5))  # not using *args


def test_getitem_setitem() -> None:
    p = InputParameters()
    p1 = InputParameter("a1", 1, label="label-1")
    p2 = InputParameter("a2", 1, label="label-2")
    p3 = InputParameter("a3", 1)
    p["a1"] = p1
    p["a2"] = p2
    p["a3"] = p3

    assert p["a1"] is p1
    assert p["label-1"] is p1
    assert p["a2"] is p2
    assert p["label-2"] is p2
    assert p["a3"] is p3

    rp = ResultParameters({"a": [-1, -2, -3], "ua": [0.1, 0.2, 0.3]}, p)
    assert rp["label-1"].name == "a1"
    assert rp["a1"].value == -1.0
    assert rp["label-1"].uncert == 0.1
    assert rp["a1"].label == "label-1"

    assert rp["label-2"].name == "a2"
    assert rp["label-2"].value == -2.0
    assert rp["a2"].uncert == 0.2
    assert rp["a2"].label == "label-2"

    assert rp["a3"].name == "a3"
    assert rp["a3"].value == -3.0
    assert rp["a3"].uncert == 0.3
    assert rp["a3"].label is None

    # cannot add a new ResultParameter to the collection
    with pytest.raises(PermissionError, match="Cannot set a ResultParameter"):
        rp["a4"] = ResultParameter("a4", 2, 1)

    # but can still modify a ResultParameter.label (the only writeable attribute)
    rp["a1"].label = "new"
    assert rp["a1"].label == "new"

    for k in ("a4", None, ""):
        with pytest.raises(KeyError, match="valid 'name' or 'label'"):
            _ = p[k]  # type: ignore[index]
        with pytest.raises(KeyError, match="valid 'name' or 'label'"):
            _ = rp[k]  # type: ignore[index]


def test_add_many() -> None:
    p = InputParameters()
    items = [InputParameter("a1", 1), ("a2", -10), {"name": "a3", "value": 100}]
    p.add_many(items)  # type: ignore[arg-type]
    assert len(p) == 3
    assert "a1" in p
    assert "a2" in p
    assert "a3" in p

    p.add_many([])
    p.add_many(())
    assert len(p) == 3

    with pytest.raises(TypeError, match="not iterable"):
        InputParameters(7)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be an iterable"):
        InputParameters([InputParameter("a1", 1), None])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="from only a string"):
        p.add_many("a1")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="from only a string"):
        InputParameters([InputParameter("a1", 1), "a1"])  # type: ignore[list-item]


def test_contains() -> None:
    p = InputParameters([InputParameter("a1", 1), ("a2", 2, True, "label"), ("a3", 3)])
    assert len(p) == 3
    assert "a1" in p
    assert "a2" in p
    assert "a3" in p
    assert "a4" not in p
    assert "label" in p

    rp = ResultParameters({"a": [11, 22, 33], "ua": [1, 2, 3]}, p)
    assert len(rp) == 3
    assert "a1" in rp
    assert "a2" in rp
    assert "a3" in rp
    assert "a4" not in rp
    assert "label" in rp


def test_pop() -> None:
    p = InputParameters([InputParameter("a1", 1), ("a2", 2, True, "label"), ("a3", 3)])
    assert len(p) == 3
    assert p.pop("a1").name == "a1"
    assert len(p) == 2
    assert p.pop("label").name == "a2"
    assert len(p) == 1
    assert p.pop("a3").value == 3.0
    with pytest.raises(KeyError, match="not a valid 'name' or 'label'"):
        p.pop("a2")


def test_repr() -> None:
    assert str(InputParameters()) == "InputParameters()"

    p = InputParameters([("a1", 1), ("a2", 2), ("a3", 3)])
    assert (
        str(p)
        == """InputParameters(
  InputParameter(name='a1', value=1.0, constant=False, label=None),
  InputParameter(name='a2', value=2.0, constant=False, label=None),
  InputParameter(name='a3', value=3.0, constant=False, label=None)
)"""
    )

    rp = ResultParameters({"a": [11, 22, 33], "ua": [1, 2, 3]}, p)
    assert (
        str(rp)
        == """ResultParameters(
  ResultParameter(name='a1', value=11.0, uncert=1.0, label=None),
  ResultParameter(name='a2', value=22.0, uncert=2.0, label=None),
  ResultParameter(name='a3', value=33.0, uncert=3.0, label=None)
)"""
    )


def test_update() -> None:
    param = InputParameter("a1", 1)
    p = InputParameters([param, ("a2", 2, True, "label"), ("a3", 3)])

    assert param.name == "a1"
    assert param.value == 1.0
    assert param.constant is False
    assert param.label is None
    assert p["a1"] is param

    p.update("a1", value=7, label="amplitude")
    assert param.name == "a1"
    assert param.value == 7.0
    assert param.constant is False
    assert param.label == "amplitude"
    # update happens in-place, a new Parameter is not created
    assert p["a1"] is param

    with pytest.raises(PermissionError, match="Cannot change InputParameter name"):
        p.update("a1", value=7, label="amplitude", name="a4")

    # the new name is the same, this is okay
    p.update("a1", value=7, label="amplitude", name="a1")

    # update based on the label
    p.update("amplitude", value=-1.1, constant=True, label=None)
    assert param.name == "a1"
    assert param.value == -1.1
    assert param.constant is True
    assert param.label is None
    # update happens in-place, a new Parameter is not created
    assert p["a1"] is param


def test_del() -> None:
    p = InputParameters([InputParameter("a1", 1), ("a2", 2, True, "label"), ("a3", 3)])
    rp = ResultParameters({"a": [2, 1, 0], "ua": [1, 2, 3]}, p)

    assert len(p) == 3
    assert len(rp) == 3

    del p["a1"]
    assert len(p) == 2
    del p["label"]
    assert len(p) == 1
    del p["a3"]
    assert len(p) == 0

    with pytest.raises(KeyError, match="not a valid 'name' or 'label'"):
        del p["a3"]

    for item in ("a1", "label", None, "", 1 + 7j):  # can be anything
        with pytest.raises(PermissionError, match=r"Cannot delete a ResultParameter"):
            del rp[item]  # type: ignore[arg-type]


def test_result_parameters() -> None:
    p = InputParameters([("a1", 1), ("a3", 3, True, "abc"), ("a2", 2)])
    rp = ResultParameters({"a": [11, 12, 13], "ua": [101, 102, 103]}, p)
    assert rp.names() == ["a1", "a2", "a3"]
    assert rp.labels() == [None, None, "abc"]
    assert np.array_equal(rp.values(), [11.0, 12.0, 13.0])
    assert np.array_equal(rp.uncerts(), [101.0, 102.0, 103.0])

    for item in (rp, list(rp)):
        for i, param in enumerate(item, start=1):
            assert param.name == f"a{i}"
            assert param.value == i + 10
            assert param.uncert == i + 100
            if i < 3:
                assert param.label is None
            else:
                assert param.label == "abc"
