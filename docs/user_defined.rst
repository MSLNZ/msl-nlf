.. _nlf-user-defined-function:

User-Defined Function
=====================
For situations when the fit equation cannot be expressed in analytical form by
using the arithmetic operations and functions that are supported, the user can
create a custom function that is compiled as a DLL. This custom DLL must export
four functions with the following names:

* GetFunctionName
* GetFunctionValue
* GetNumParameters
* GetNumVariables

How to define these four functions is best shown with examples.

.. _nlf-user-defined-cpp1:

C++ Example (1D)
----------------
A user-defined function is created in C++ in order to fit the Roszman1_ dataset
that is provided by NIST. This fit equation requires the *arctan* function,
which is not one of the built-in functions that are currently supported by the
Delphi software (but could be upon request).

The header file is

.. literalinclude:: _static/Roszman1.h
   :language: c++

and the source file is

.. literalinclude:: _static/Roszman1.cpp
   :language: c++

To compile the C++ source code to a DLL, one could use `Visual Studio C++`_,

.. code-block:: console

    cl.exe /LD Roszman1.cpp


.. _nlf-user-defined-cpp2:

C++ Example (2D)
----------------
A user-defined function is created in C++ in order to fit the Nelson_ dataset
that is provided by NIST. This fit equation, **a1-a2*x1*exp(-a3*x2)**, could
have been passed directly to a :class:`~msl.nlf.model.Model` since all
arithmetic operations and functions are supported; however, this example
illustrates how to handle situations when there are multiple :math:`x` variables

The header file is

.. literalinclude:: _static/Nelson.h
   :language: c++

and the source file is

.. literalinclude:: _static/Nelson.cpp
   :language: c++

To compile the C++ source code to a DLL, one could use `Visual Studio C++`_,

.. code-block:: console

    cl.exe /LD Nelson.cpp

.. _nlf-user-defined-delphi:

Delphi Example
--------------
A user-defined function is created in Delphi Pascal for the Beta Distribution.

.. literalinclude:: _static/beta.dpr
   :language: pascal

.. _nlf-user-usage:

Using the Function
------------------
To use a custom function, the first parameter passed when defining a
:class:`~msl.nlf.model.Model` must be the first part of the name, up to the colon,
defined in *GetFunctionName*, and, optionally, specify the directory where the
custom DLL is located as a *user_dir* keyword argument. If you are also using the
Delphi GUI, the directory that has been set in the GUI for the user-defined functions
will be used as the default *user_dir* value. Otherwise, the current working directory
is used as the default *user_dir* value if a directory is not explicitly specified.

Below, the C++ function, *f1*, is used as the custom function

.. code-block:: python

    from msl.nlf import Model

    model = Model('f1', user_dir='./tests/user_defined')


.. _Roszman1: https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Roszman1.dat
.. _Nelson: https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Nelson.dat
.. _Visual Studio C++: https://visualstudio.microsoft.com/vs/features/cplusplus/
