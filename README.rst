=======
MSL-NLF
=======

|docs| |pypi| |github tests|

MSL-NLF is a wrapper around the non-linear fitting software written by
P. Saunders at the Measurement Standards Laboratory (MSL) of New Zealand. The
original source code is written in Delphi and compiled to a shared library
(DLL) to be accessed by other programming languages, such as Python. The
Delphi software also provides a GUI to visualize and interact with the fitted
data. Please contact someone from MSL if you are interested in using the GUI
version.

Install
-------
The DLL files have been compiled for use on Windows and therefore only
Windows is supported.

To install MSL-NLF run:

.. code-block:: console

   pip install msl-nlf

Alternatively, using the `MSL Package Manager`_ run:

.. code-block:: console

   msl install nlf

Documentation
-------------
The documentation for MSL-NLF can be found here_.

Quick Start
-----------
As a simple example, one might need to model data that has a linear relationship

.. code-block:: pycon

   >>> x = [1.6, 3.2, 5.5, 7.8, 9.4]
   >>> y = [7.8, 19.1, 17.6, 33.9, 45.4]

The first task to perform is to create a Model_ and specify the fit equation as
a string (see the documentation of Model_ for an overview of what arithmetic
operations and functions are allowed in the equation)

.. code-block:: pycon

   >>> from msl.nlf import Model
   >>> model = Model('a1+a2*x')

Provide an initial guess for the parameters (*a1* and *a2*) and apply the fit

.. code-block:: pycon

   >>> result = model.fit(x, y, params=[1, 1])
   >>> result.params
   ResultParameters(
     ResultParameter(name='a1', value=0.522439024..., uncert=5.132418149..., label=None),
     ResultParameter(name='a2', value=4.406829268..., uncert=0.827701724..., label=None)
   )

The *result* object that is returned contains information about the fit result,
such as the chi-square value and the covariance matrix, but we simply showed
a summary of the fit parameters above.

If you want to have control over which parameters should be held constant during the
fitting process and which are allowed to vary or if you want to assign a label to a
parameter, you need to create an InputParameters_ instance.

In this case, we will use one of the built-in models to perform a linear fit and
create InputParameters_. We use the InputParameters_ instance to provide an initial
value for each parameter, define labels, and set whether the initial value of a
parameter is held constant during the fitting process

.. code-block:: pycon

   >>> from msl.nlf import LinearModel
   >>> model = LinearModel()
   >>> model.equation
   'a1+a2*x'
   >>> params = model.create_parameters()
   >>> a1 = params.add(name='a1', value=0, constant=True, label='intercept')
   >>> a2 = params.add('a2', 4, False, 'slope')  # alternative way to add a parameter
   >>> result = model.fit(x, y, params=params)
   >>> result.params
   ResultParameters(
      ResultParameter(name='a1', value=0.0, uncert=5.1324181499..., label='intercept'),
      ResultParameter(name='a2', value=4.4815604681..., uncert=0.3315980376..., label='slope')
   )


.. |docs| image:: https://readthedocs.org/projects/msl-nlf/badge/?version=latest
   :target: https://msl-nlf.readthedocs.io/en/latest/
   :alt: Documentation Status

.. |pypi| image:: https://badge.fury.io/py/msl-nlf.svg
   :target: https://badge.fury.io/py/msl-nlf

.. |github tests| image:: https://github.com/MSLNZ/msl-nlf/actions/workflows/run-tests.yml/badge.svg
   :target: https://github.com/MSLNZ/msl-nlf/actions/workflows/run-tests.yml

.. _MSL Package Manager: https://msl-package-manager.readthedocs.io/en/stable/
.. _here: https://msl-nlf.readthedocs.io/en/latest/index.html
.. _Model: https://msl-nlf.readthedocs.io/en/latest/_api/msl.nlf.model.html#msl.nlf.model.Model
.. _InputParameters: https://msl-nlf.readthedocs.io/en/latest/_api/msl.nlf.parameter.html#msl.nlf.parameter.InputParameters
