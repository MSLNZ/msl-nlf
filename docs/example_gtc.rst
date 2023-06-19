.. _nlf-example-gtc:

============================
Uncertain Real Numbers (GTC)
============================
An example from Appendix H3 of the GUM [#]_.

This example requires GTC_ to be installed, to install it run

.. code-block:: console

    pip install GTC

As defined in Appendix H3 of the GUM, the calibration curve is

.. math::

   b(t) = y_1 + y_2 (t-t_0)

where the reference temperature, :math:`t_0`, is chosen to be 20 :math:`^\circ C`.

This translates to the following equation that is passed to a
:class:`~msl.nlf.model.Model`

.. math::

   f(x; a) = a_1 + a_2 (x-20)

The *intercept* (:math:`a_1`) and *slope* (:math:`a_2`) result parameters
are converted to a correlated ensemble of
:ref:`uncertain real numbers <uncertain_real_number>` (via the
:meth:`~msl.nlf.datatypes.Result.to_ureal` method) which are
used to calculate the response at a chosen stimulus.

.. code-block:: python

    from msl.nlf import Model

    # Thermometer readings (degrees C)
    x = (21.521, 22.012, 22.512, 23.003, 23.507, 23.999, 24.513, 25.002, 25.503, 26.010, 26.511)

    # Observed differences with calibration standard (degrees C)
    y = (-0.171, -0.169, -0.166, -0.159, -0.164, -0.165, -0.156, -0.157, -0.159, -0.161, -0.160)

    # Arbitrary offset temperature (degrees C)
    t0 = 20

    # Create the model
    model = Model(f'a1+a2*(x-{t0})')

    # Create an initial guess. Allow the intercept and slope to vary during
    # the fitting process and assign helpful labels
    params = model.create_parameters([
        ('a1', 1, False, 'intercept'),
        ('a2', 1, False, 'slope')])

    # Apply the fit
    result = model.fit(x, y, params=params)

    # Convert the result to a correlated ensemble of uncertain real numbers
    intercept, slope = result.to_ureal()

The *intercept* and *slope* can be used to calculate a correction for
a reading of 30 :math:`^\circ C`

.. code-block:: console

    >>> intercept + slope*(30 - t0)
    ureal(-0.14937681268874...,0.004138595752854...,9.0)


.. _GTC: https://gtc.readthedocs.io/en/stable/

.. [#]
    BIPM and IEC and IFCC and ISO and IUPAC and IUPAP and OIML,
    *Evaluation of measurement data - Guide to the expression of uncertainty in measurement JCGM 100:2008 (GUM 1995 with minor corrections)*, (2008)
    `http://www.bipm.org/en/publications/guides/gum <http://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e.html?csnumber=50461>`_
