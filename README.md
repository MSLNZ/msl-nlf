# MSL-NLF

[![Tests Status](https://github.com/MSLNZ/msl-nlf/actions/workflows/tests.yml/badge.svg)](https://github.com/MSLNZ/msl-nlf/actions/workflows/tests.yml)
[![Docs Status](https://github.com/MSLNZ/msl-nlf/actions/workflows/docs.yml/badge.svg)](https://github.com/MSLNZ/msl-nlf/actions/workflows/docs.yml)

## Overview
MSL-NLF is a Python wrapper around the non-linear fitting software that was written by P. Saunders at the Measurement Standards Laboratory (MSL) of New Zealand. The original source code is written in Delphi and compiled to a shared library to be accessed by other programming languages. The Delphi software also provides a GUI to visualize and interact with the fitted data. Please contact someone from MSL if you are interested in using the GUI version.

## Install
`msl-nlf` is available for installation via the [Python Package Index](https://pypi.org/){:target="_blank"} and may be installed with [pip](https://pip.pypa.io/en/stable/){:target="_blank"}

```console
pip install msl-nlf
```

## Documentation
The documentation for `msl-nlf` is available [here](https://mslnz.github.io/msl-nlf/){:target="_blank"}.

## Quick Start
As a simple example, one might need to model data that has a linear relationship

```python
>>> x = [1.6, 3.2, 5.5, 7.8, 9.4]
>>> y = [7.8, 19.1, 17.6, 33.9, 45.4]
```

The first task to perform is to create a [Model]{:target="_blank"} and specify the fit equation as a string (see the documentation of [Model]{:target="_blank"} for an overview of what arithmetic operations and functions are allowed in the equation)

```python
>>> from msl.nlf import Model
>>> model = Model("a1+a2*x")

```

Provide an initial guess for the parameters (*a1* and *a2*) and apply the fit

```python
>>> result = model.fit(x, y, params=[1, 1])
>>> result.params
ResultParameters(
  ResultParameter(name='a1', value=0.522439024..., uncert=5.132418149..., label=None),
  ResultParameter(name='a2', value=4.406829268..., uncert=0.827701724..., label=None)
)

```

The [Result]{:target="_blank"} object that is returned contains information about the fit result, such as the chi-square value and the covariance matrix, but we simply showed a summary of the fit parameters above.

If you want to have control over which parameters should be held constant during the fitting process and which are allowed to vary or if you want to assign a label to a parameter, you need to create an [InputParameters]{:target="_blank"} instance.

In this case, we will use one of the built-in [models]{:target="_blank"} to perform a linear fit and create [InputParameters]{:target="_blank"}. We use the [InputParameters]{:target="_blank"} instance to provide an initial value for each parameter, define labels, and set whether the initial value of a parameter is held constant during the fitting process

```python
>>> from msl.nlf import LinearModel
>>> model = LinearModel()
>>> model.equation
'a1+a2*x'
>>> params = model.create_parameters()
>>> a1 = params.add(name="a1", value=0, constant=True, label="intercept")
>>> params["a2"] = 4, False, "slope"  # alternative way to add a parameter
>>> result = model.fit(x, y, params=params)
>>> result.params
ResultParameters(
  ResultParameter(name='a1', value=0.0, uncert=0.0, label='intercept'),
  ResultParameter(name='a2', value=4.4815604681..., uncert=0.3315980376..., label='slope')
)

```

[Model]: https://mslnz.github.io/msl-nlf/api/model/#msl.nlf.model.Model
[InputParameters]: https://mslnz.github.io/msl-nlf/api/parameters/#msl.nlf.parameters.InputParameters
[Result]: https://mslnz.github.io/msl-nlf/api/datatypes/#msl.nlf.datatypes.Result
[models]: https://mslnz.github.io/msl-nlf/api/models/
