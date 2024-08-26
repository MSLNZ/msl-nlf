# Overview
`msl-nlf` is a Python wrapper for the non-linear fitting software that was written by Peter Saunders at the Measurement Standards Laboratory (MSL) of New Zealand. The original source code is written in Delphi and compiled to a shared library to be accessed by other programming languages. The Delphi software also provides a GUI (for Windows only) to visualize and interact with the data. Please [contact](https://www.measurement.govt.nz/contact-us-2/){:target="_blank"} MSL if you are interested in using the GUI version.

The application of non-linear fitting is a general-purpose, curving-fitting program that will fit any curve of the form

$$
y = f(x_1, x_2, ..., x_n, a_1, a_2, ..., a_N)
$$

to a set of data, where $x_1, x_2, ..., x_n$, are real variables, which may or may not be correlated, and $a_1, a_2, ..., a_N$ are real parameters. In the Delphi algorithm, $n$ can be any value from 1 to 30 and $N$ from 1 to 99. The function $f$ will be fitted to a set of $M$ data points, $(x_{1,1}, x_{2,1}, ..., x_{n,1}, y_1), (x_{1,2}, x_{2,2}, ..., x_{n,2}, y_2), ..., (x_{1,M}, x_{2,M}, ..., x_{n,M}, y_M)$, where $M \geq N$. The parameters $a_1, a_2, ..., a_N$ can be chosen to be either constant or fitted, providing additional flexibility to the fitting process. If there are constant parameters, then $M$ must be greater than or equal to the number of non-constant parameters.

For more details see *Propagation of uncertainty for non-linear calibration equations with an application in radiation thermometry*, **Peter Saunders**, [Metrologia 40 93 (2003)](https://doi.org/10.1088/0026-1394/40/2/315){:target="_blank"}.

The non-linear fitting algorithm implements the following features:

* perform an unweighted fit or a weighted fit with uncertainties in the $x$ and/or $y$ data
* setting correlations between the $x_i-x_i$, $x_i-x_j$, $x_i-y$ and/or $y-y$ data
* whether second-derivative terms (Hessian) are included in the calculation
* use up to 30 independent variables and up to 99 parameters in the fit equation
* whether a parameter is held constant or allowed to vary during the fitting process
* choice of different fitting methods (see the [FitMethod][msl.nlf.datatypes.FitMethod] enum)

Follow the [Install][install] instructions and read the [Getting Started][getting-started] guide to begin.
