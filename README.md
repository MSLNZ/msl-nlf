# MSL-NLF

[![Tests Status](https://github.com/MSLNZ/msl-nlf/actions/workflows/tests.yml/badge.svg)](https://github.com/MSLNZ/msl-nlf/actions/workflows/tests.yml)
[![Docs Status](https://github.com/MSLNZ/msl-nlf/actions/workflows/docs.yml/badge.svg)](https://github.com/MSLNZ/msl-nlf/actions/workflows/docs.yml)

## Overview
MSL-NLF is a Python wrapper around the non-linear fitting software that was written by P. Saunders at the Measurement Standards Laboratory (MSL) of New Zealand. The original source code is written in Delphi and compiled to a shared library to be accessed by other programming languages. The Delphi software also provides a GUI to visualize and interact with the fitted data. Please contact someone from MSL if you are interested in using the GUI version.

## Install
`msl-nlf` is available for installation via the [Python Package Index](https://pypi.org/) and may be installed with [pip](https://pip.pypa.io/en/stable/)

```console
pip install msl-nlf
```

## Documentation
The documentation for `msl-nlf` is available [here](https://mslnz.github.io/msl-nlf/).

## Quick Start
As a simple example, one might need to model data that has a linear relationship

```pycon
>>> x = [1.6, 3.2, 5.5, 7.8, 9.4]
>>> y = [7.8, 19.1, 17.6, 33.9, 45.4]
```

The first task to perform is to create a [Model] and specify the fit equation as a string (see the documentation of [Model] for an overview of what arithmetic operations and functions are allowed in the equation)

```pycon
>>> from msl.nlf import Model
>>> model = Model('a1+a2*x')
```

Provide an initial guess for the parameters (*a1* and *a2*) and apply the fit

```pycon
>>> result = model.fit(x, y, params=[1, 1])
>>> result.params
ResultParameters(
   ResultParameter(name='a1', value=0.522439024..., uncert=5.132418149..., label=None),
   ResultParameter(name='a2', value=4.406829268..., uncert=0.827701724..., label=None)
)
```

The [Result] object that is returned contains information about the fit result, such as the chi-square value and the covariance matrix, but we simply showed a summary of the fit parameters above.

If you want to have control over which parameters should be held constant during the fitting process and which are allowed to vary or if you want to assign a label to a parameter, you need to create an [InputParameters] instance.

In this case, we will use one of the built-in [models] to perform a linear fit and create [InputParameters]. We use the [InputParameters] instance to provide an initial value for each parameter, define labels, and set whether the initial value of a parameter is held constant during the fitting process

```pycon
>>> from msl.nlf import LinearModel
>>> model = LinearModel()
>>> model.equation
'a1+a2*x'
>>> params = model.create_parameters()
>>> a1 = params.add(name='a1', value=0, constant=True, label='intercept')
>>> params['a2'] = 4, False, 'slope'  # alternative way to add a parameter
>>> result = model.fit(x, y, params=params)
>>> result.params
ResultParameters(
   ResultParameter(name='a1', value=0.0, uncert=0.0, label='intercept'),
   ResultParameter(name='a2', value=4.4815604681..., uncert=0.3315980376..., label='slope')
)
```

[Model]: https://msl-nlf.readthedocs.io/en/latest/_api/msl.nlf.model.html#msl.nlf.model.Model
[InputParameters]: https://msl-nlf.readthedocs.io/en/latest/_api/msl.nlf.parameter.html#msl.nlf.parameter.InputParameters
[Result]: https://msl-nlf.readthedocs.io/en/latest/_api/msl.nlf.datatypes.html#msl.nlf.datatypes.Result
[models]: https://msl-nlf.readthedocs.io/en/latest/_api/msl.nlf.models.html
