# Compiled (User-defined) Function

## Overview
For situations when the fit equation cannot be expressed in analytical form by using the arithmetic operations and functions that are supported, a user-defined function that is compiled as a shared library may be used. This compiled function must export four functions with the following names:

* GetFunctionName
* GetFunctionValue
* GetNumParameters
* GetNumVariables

## Examples
How to define these four functions is best shown with examples.

### C++ (1D data)
A compiled function is created in C++ in order to fit the [Roszman1]{:target="_blank"} dataset that is provided by NIST. This fit equation requires the *arctan* function, which is not one of the built-in functions that are currently supported by the Delphi software (but could be).

The header file is

```cpp
--8<-- "docs/assets/Roszman1.h"
```

and the source file is

```cpp
--8<-- "docs/assets/Roszman1.cpp"
```

To compile the C++ source code to a shared library, one could use [Visual Studio C++]{:target="_blank"}

```console
cl /LD Roszman1.cpp
```

or [gcc]{:target="_blank"}

```console
gcc -shared Roszman1.cpp
```

### C++ (2D data)
A compiled function is created in C++ in order to fit the [Nelson]{:target="_blank"} dataset that is provided by NIST. This fit equation, `a1-a2*x1*exp(-a3*x2)`, could have been passed directly to a [Model][msl.nlf.model.Model] since all arithmetic operations and functions are supported; however, this example illustrates how to define a function if multiple $x$ variables are required.

The header file is

```cpp
--8<-- "docs/assets/Nelson.h"
```

and the source file is

```cpp
--8<-- "docs/assets/Nelson.cpp"
```

To compile the C++ source code to a shared library, one could use [Visual Studio C++]{:target="_blank"}

```console
cl /LD Nelson.cpp
```

or [gcc]{:target="_blank"}

```console
gcc -shared Nelson.cpp
```

### Delphi
A compiled function is created in Delphi Pascal for the Beta Distribution.

```pascal
--8<-- "docs/assets/beta.dpr"
```

## Using the Compiled Function
To use a compiled function, the `equation` parameter when defining a [Model][msl.nlf.model.Model] must be the first part of the function name (up to, but excluding, the colon) defined in *GetFunctionName*, and, optionally, specify the directory where the function is located as a `user_dir` keyword argument. If you are also using the Delphi GUI, the directory that has been set in the GUI for the user-defined functions will be used as the default `user_dir` value. Otherwise, the current working directory is used as the default `user_dir` value if a directory is not explicitly specified.

Below, the C++ function `f1` (from the [example][c-1d-data]) is used for the [Model][msl.nlf.model.Model]

```python
from msl.nlf import Model

model = Model("f1", user_dir="./tests/user_defined")
```

## 32-bit Compiled Function
A 32-bit version of the Delphi shared library may be loaded in 64-bit Python (Windows only). The primary reason for using the 32-bit shared library on Windows (by setting `win32=True` when creating a [Model][msl.nlf.model.Model]) is if you have user-defined functions that were compiled to a 32-bit DLL for use with the 32-bit Delphi GUI application and you do not have a 64-bit version of the DLL. The following table illustrates the differences between using a compiled function as a 32-bit DLL and a 64-bit DLL.

|         <center>32-bit DLL</center>                       |        <center>64-bit DLL</center>         |
| --------------------------------------------------------- | ------------------------------------------ |
| Can be used in both 32- and 64-bit versions of Python     | Can only be used in 64-bit Python          |
| When used in 64-bit Python, the fit will take longer [^1] | There is no performance overhead           |
| Limited to 4GB RAM                                        | Can access more than 4GB RAM               |

If loading the 32-bit DLL in 64-bit Python, it is important to reduce the number of times a [Model][msl.nlf.model.Model] is created to fit data. In this case, creating a [Model][msl.nlf.model.Model] object takes about 1 second for a client-server protocol to be initialized in the background. Once the [Model][msl.nlf.model.Model] has been created, the client and server are running and repeatedly calling the [fit][msl.nlf.model.Model.fit] method will be more efficient (but still slower than fitting data with the 64-bit DLL in 64-bit Python, or the 32-bit DLL in 32-bit Python).

Pseudocode is shown below that demonstrates the recommended way to apply fits if loading the 32-bit DLL in 64-bit Python. See [A Model as a Context Manager][a-model-as-a-context-manager] for more details about the use of the *with* statement

```
# Don't do this. Don't create a new model to process each data file.
for data in data_files:
    with LinearModel(win32=True) as model:
        result = model.fit(data.x, data.y)

# Do this instead. Create a model once and then fit each data file.
with LinearModel(win32=True) as model:
    for data in data_files:
        result = model.fit(data.x, data.y)
```

[^1]: This is not due to the 32-bit Delphi code, but due to an overhead on the Python side to exchange data between 64-bit Python and a 32-bit DLL. When the 32-bit DLL is used in 32-bit Python, there is no overhead.

[Roszman1]: https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Roszman1.dat
[Nelson]: https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Nelson.dat
[Visual Studio C++]: https://visualstudio.microsoft.com/vs/features/cplusplus/
[gcc]: https://gcc.gnu.org/
