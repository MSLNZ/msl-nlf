.. _nlf-32vs64:

32-bit vs 64-bit DLL
====================
A 32-bit and a 64-bit version of the DLL are provided. The following table
illustrates the differences between the DLL versions.

.. table:: 32-bit vs 64-bit comparison
    :widths: 50 50
    :align: center

    =========================================================  ===============================================
                           32-bit DLL                                              64-bit DLL
    =========================================================  ===============================================
    Can be used in both 32- and 64-bit versions of Python      Can only be used in 64-bit Python
    When used in 64-bit Python, the fit will take longer [#]_  There is no performance overhead
    Delphi uses 10 bytes for the floating-point type           Delphi uses 8 bytes for the floating-point type
    Limited to 4GB RAM                                         Can access more than 4GB RAM
    =========================================================  ===============================================

If loading the 32-bit DLL in 64-bit Python, it is important to reduce the number
of times a :class:`~msl.nlf.model.Model` is created to fit data. In this case,
creating a :class:`~msl.nlf.model.Model` object takes about 1 second for a
client-server protocol to be initialized in the background. Once the
:class:`~msl.nlf.model.Model` has been created, the client and server are running
and repeatedly calling the :meth:`~msl.nlf.model.Model.fit` method will be more
efficient (but still slower than fitting data with the 64-bit DLL in 64-bit Python,
or the 32-bit DLL in 32-bit Python).

Pseudocode is shown below that demonstrates the best way to apply fits if
loading the 32-bit DLL in 64-bit Python. See :ref:`nlf-context-manager`
for more details about the use of the *with* statement::

    # Don't do this. Don't create a new model to process each data file.
    for data in data_files:
        with LinearModel(dll='nlf32') as model:
            result = model.fit(data.x, data.y)

    # Do this instead. Create a model once and then fit each data file.
    with LinearModel(dll='nlf32') as model:
        for data in data_files:
            result = model.fit(data.x, data.y)


.. [#]
    This is not due to the 32-bit Delphi code, but due to an overhead on the
    Python side to exchange data between 64-bit Python and a 32-bit DLL.
    When the 32-bit DLL is used in 32-bit Python, there is no overhead.
