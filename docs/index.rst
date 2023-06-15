.. _msl-nlf-welcome:

=======
MSL-NLF
=======
MSL-NLF is a Python wrapper around the non-linear fitting software that was
written by P. Saunders at the Measurement Standards Laboratory (MSL) of New
Zealand. The original source code is written in Delphi and compiled to a shared
library (DLL) to be accessed by other programming languages. The Delphi software
also provides a GUI to visualize and interact with the fitted data. Please
contact someone from MSL if you are interested in using the GUI version.

The application of non-linear fitting is a general-purpose curving-fitting
program that will fit any curve of the form

.. math::

   y = f(x_1, x_2, ..., x_n, a_1, a_2, ..., a_N),

to a set of data, where :math:`x_1, x_2, ..., x_n`, are real variables, which
may or may not be correlated, and :math:`a_1, a_2, ..., a_N` are real parameters.
In the Delphi algorithm, :math:`n` can be any value from 1 to 30 and :math:`N`
from 1 to 99. The function :math:`f` will be fitted to a set of :math:`M` data
points, :math:`(x_{1,1}, x_{2,1}, ..., x_{n,1}, y_1),`
:math:`(x_{1,2}, x_{2,2}, ..., x_{n,2}, y_2), ...,`
:math:`(x_{1,M}, x_{2,M}, ..., x_{n,M}, y_M)`,
where :math:`M \geq N`. The parameters :math:`a_1, a_2, ..., a_N` can be chosen
to be either constant or fitted, providing additional flexibility to the fitting
process. If there are constant parameters, then :math:`M` must be greater than
or equal to the number of non-constant parameters.

For more details see *Propagation of uncertainty for non-linear calibration*
*equations with an application in radiation thermometry*, **P. Saunders**,
`Metrologia 40 93 (2003) <https://doi.org/10.1088/0026-1394/40/2/315>`_.

The non-linear fitting algorithm implements the follow features:

1. perform an unweighted fit or a weighted fit with uncertainties in the :math:`x` and/or :math:`y` data
2. setting correlations between the :math:`x_i-x_i`, :math:`x_i-x_j`, :math:`x_i-y` and/or :math:`y-y` data
3. whether second-derivative terms (Hessian) are included in the calculation
4. use up to 30 independent variables and up to 99 parameters in the fit equation
5. whether a parameter is held constant or allowed to vary during the fitting process
6. choice of different fitting methods (see the :class:`~msl.nlf.datatypes.FitMethod` enum)

Follow the :ref:`nlf-install` instructions and read the :ref:`nlf-getting-started`
guide to begin.

Contents
========

.. toctree::
   :maxdepth: 1

   install
   getting_started
   examples
   32-bit vs 64-bit DLL <32vs64bit>
   API <api>
   License <license>
   Authors <authors>
   Release Notes <changelog>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
