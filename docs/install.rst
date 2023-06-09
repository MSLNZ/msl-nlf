.. _nlf-install:

=======
Install
=======

.. note::

    A PyPI release is not available yet. You can install from the main branch for now.

The DLL files have been compiled for use on Windows and therefore only
Windows is supported.

To install MSL-NLF run

.. code-block:: console

   pip install msl-nlf

Alternatively, using the `MSL Package Manager`_ run

.. code-block:: console

   msl install nlf

.. _nlf-dependencies:

Dependencies
------------
* Python 3.8+
* numpy_

Optional Dependencies
+++++++++++++++++++++
The GUM Tree Calculator, GTC_, is not automatically installed when MSL-NLF
is installed, but it is required to create a correlated ensemble of
:ref:`uncertain real numbers <uncertain_real_number>`
from a :class:`~msl.nlf.datatypes.Result`.

To automatically include GTC_ when installing MSL-NLF run

.. code-block:: console

   pip install msl-nlf[gtc]

.. _MSL Package Manager: https://msl-package-manager.readthedocs.io/en/stable/
.. _numpy: https://www.numpy.org/
.. _GTC: https://gtc.readthedocs.io/en/stable/
