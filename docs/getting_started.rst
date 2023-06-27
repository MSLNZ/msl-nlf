.. _nlf-getting-started:

===============
Getting Started
===============
As a simple example, one might need to model data that has a linear relationship

.. code-block:: pycon

    >>> x = [1.6, 3.2, 5.5, 7.8, 9.4]
    >>> y = [7.8, 19.1, 17.6, 33.9, 45.4]

The first task to perform is to create a :class:`~msl.nlf.model.Model` and specify
the fit equation as a string (see the documentation of :class:`~msl.nlf.model.Model`
for an overview of what arithmetic operations and functions are allowed in the equation)

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

The :class:`~msl.nlf.datatypes.Result` object that is returned from the fit contains
information about the fit result, such as the chi-square value and the covariance matrix,
but we simply showed a summary of the fit parameters above.

.. _nlf-input-parameters:

Input Parameters
----------------
If you want to have control over which parameters should be held constant during the
fitting process and which are allowed to vary or if you want to assign a label to a
parameter, you need to create an :class:`~msl.nlf.parameter.InputParameters` instance.

In this case, we will use one of the built-in :ref:`models <nlf-models>`,
:class:`~msl.nlf.models.LinearModel`, to perform the linear fit and create
:class:`~msl.nlf.parameter.InputParameters`. We use the
:class:`~msl.nlf.parameter.InputParameters` instance to provide an initial
value for each parameter, define labels, and set whether the initial value of a
parameter is held constant during the fitting process

.. code-block:: pycon

    >>> from msl.nlf import LinearModel
    >>> model = LinearModel()
    >>> model.equation
    'a1+a2*x'
    >>> params = model.create_parameters()
    >>> a1 = params.add(name='a1', value=0, constant=True, label='intercept')
    >>> params['a2'] = 1, False, 'slope'  # alternative way to add a parameter
    >>> result = model.fit(x, y, params=params)
    >>> result.params
    ResultParameters(
      ResultParameter(name='a1', value=0.0, uncert=0.0, label='intercept'),
      ResultParameter(name='a2', value=4.4815604681..., uncert=0.3315980376..., label='slope')
    )

We showed above that calling :meth:`~msl.nlf.model.Model.create_parameters` is
one way to create an :class:`~msl.nlf.parameter.InputParameters` instance. It
can also be instantiated directly

.. code-block:: pycon

    >>> from msl.nlf import InputParameters
    >>> params = InputParameters()

There are multiple ways to add a parameter to an
:class:`~msl.nlf.parameter.InputParameters` object. To add a parameter, you
could explicitly add an instance of an :class:`~msl.nlf.parameter.InputParameter`
(using the :meth:`~msl.nlf.parameter.InputParameters.add` method or as one would
add items to a :class:`dict`)

.. code-block:: pycon

    >>> from msl.nlf import InputParameter
    >>> a1 = params.add(InputParameter('a1', 1))
    >>> a2 = params.add(InputParameter('a2', 2, constant=True))
    >>> a3 = params.add(InputParameter('a3', 3, constant=True, label='label-3'))
    >>> params['a4'] = InputParameter('a4', 4)

You could also specify positional arguments (or set it equal to a :class:`tuple`)

.. code-block:: pycon

    >>> a5 = params.add('a5', 5)
    >>> a6 = params.add('a6', 6, True)
    >>> a7 = params.add('a7', 7, False, 'label-7')
    >>> params['a8'] = 8
    >>> params['a9'] = 9, True
    >>> params['a10'] = 10, True, 'label-10'

or you could specify keyword arguments (or set it equal to a :class:`dict`)

.. code-block:: pycon

    >>> a11 = params.add(name='a11', value=11)
    >>> a12 = params.add(name='a12', value=12, constant=True)
    >>> a13 = params.add(name='a13', value=13, label='label-13')
    >>> a14 = params.add(name='a14', value=14, constant=False, label='label-14')
    >>> params['a15'] = {'value': 15}
    >>> params['a16'] = {'value': 16, 'constant': True}
    >>> params['a17'] = {'value': 17, 'label': 'label-17'}
    >>> params['a18'] = {'value': 18, 'constant': False, 'label': 'label-18'}

There is an :meth:`~msl.nlf.parameter.InputParameters.add_many` method as well.

Here, we iterate through the collection of input parameters to see what it contains

.. code-block:: pycon

    >>> for param in params:
    ...     print(param)
    InputParameter(name='a1', value=1.0, constant=False, label=None)
    InputParameter(name='a2', value=2.0, constant=True, label=None)
    InputParameter(name='a3', value=3.0, constant=True, label='label-3')
    InputParameter(name='a4', value=4.0, constant=False, label=None)
    InputParameter(name='a5', value=5.0, constant=False, label=None)
    InputParameter(name='a6', value=6.0, constant=True, label=None)
    InputParameter(name='a7', value=7.0, constant=False, label='label-7')
    InputParameter(name='a8', value=8.0, constant=False, label=None)
    InputParameter(name='a9', value=9.0, constant=True, label=None)
    InputParameter(name='a10', value=10.0, constant=True, label='label-10')
    InputParameter(name='a11', value=11.0, constant=False, label=None)
    InputParameter(name='a12', value=12.0, constant=True, label=None)
    InputParameter(name='a13', value=13.0, constant=False, label='label-13')
    InputParameter(name='a14', value=14.0, constant=False, label='label-14')
    InputParameter(name='a15', value=15.0, constant=False, label=None)
    InputParameter(name='a16', value=16.0, constant=True, label=None)
    InputParameter(name='a17', value=17.0, constant=False, label='label-17')
    InputParameter(name='a18', value=18.0, constant=False, label='label-18')

or just get all of the values

.. code-block:: pycon

    >>> params.values()
    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10.,  11.,  12.,  13.,
           14., 15., 16., 17., 18.])

You can get a specific parameter by its *name* or *label* (provide that the
*label* is not :data:`None`)

.. code-block:: pycon

    >>> params['a3']
    InputParameter(name='a3', value=3.0, constant=True, label='label-3')
    >>> params['label-14']
    InputParameter(name='a14', value=14.0, constant=False, label='label-14')

and you can update a parameter by specifying its *name* or *label* to the
:meth:`~msl.nlf.parameter.InputParameters.update` method

.. code-block:: pycon

    >>> params.update('a1', value=5.3, label='intercept')
    >>> params['a1']
    InputParameter(name='a1', value=5.3, constant=False, label='intercept')

    >>> params.update('label-7', value=1e3, constant=True, label='amplitude')
    >>> params['a7']
    InputParameter(name='a7', value=1000.0, constant=True, label='amplitude')

or you can update a parameter by directly modifying an attribute

.. code-block:: pycon

    >>> a1.label = 'something-new'
    >>> a1.constant = False
    >>> a1.value = -3.2
    >>> params['a1']
    InputParameter(name='a1', value=-3.2, constant=False, label='something-new')

    >>> params['label-3'].label = 'fwhm'
    >>> params['fwhm'].constant = True
    >>> params['fwhm'].value = 0.03
    >>> params['a3']
    InputParameter(name='a3', value=0.03, constant=True, label='fwhm')

.. _nlf-debugging:

Debugging (Input)
-----------------
If you call the :meth:`~msl.nlf.model.Model.fit` method with *debug=True* the
fit function in the DLL is not called and an :class:`~msl.nlf.datatypes.Input`
object is returned that contains the information that would have been sent
to the fit function in the DLL

.. code-block:: pycon

    >>> model = LinearModel()
    >>> info = model.fit(x, y, params=[1, 1], debug=True)
    >>> info.weighted
    False
    >>> info.fit_method
    <FitMethod.LM: 'Levenberg-Marquardt'>
    >>> info.x
    array([[1.6, 3.2, 5.5, 7.8, 9.4]])

You can display a summary of the input information

    >>> info
    Input(
      absolute_residuals=True
      correlated=False
      correlations=
        Correlations(
          data=[]
          is_correlated=[[False False]
                         [False False]]
        )
      delta=0.1
      equation='a1+a2*x'
      fit_method=<FitMethod.LM: 'Levenberg-Marquardt'>
      max_iterations=999
      params=
        InputParameters(
          InputParameter(name='a1', value=1.0, constant=False, label=None),
          InputParameter(name='a2', value=1.0, constant=False, label=None)
        )
      residual_type=<ResidualType.DY_X: 'dy v x'>
      second_derivs_B=True
      second_derivs_H=True
      tolerance=1e-20
      ux=[[0. 0. 0. 0. 0.]]
      uy=[0. 0. 0. 0. 0.]
      uy_weights_only=False
      weighted=False
      x=[[1.6 3.2 5.5 7.8 9.4]]
      y=[ 7.8 19.1 17.6 33.9 45.4]
    )

.. _nlf-fit-result:

Fit Result
----------
When a fit is performed, the returned object is a
:class:`~msl.nlf.datatypes.Result` instance

.. code-block:: pycon

    >>> model = LinearModel()
    >>> result = model.fit(x, y, params=[1, 1])
    >>> result.chisq
    84.266087804...
    >>> result.correlation
    array([[ 1.        , -0.88698141],
           [-0.88698141,  1.        ]])
    >>> result.params.values()
    array([0.52243902, 4.40682927])
    >>> for param in result.params:
    ...     print(param.name, param.value, param.uncert)
    a1 0.5224390243941... 5.132418149940...
    a2 4.4068292682920... 0.827701724508...

You can display a summary of the fit result

.. code-block:: pycon

    >>> result
    Result(
      calls=2
      chisq=84.266087804878
      correlation=[[ 1.         -0.88698141]
                   [-0.88698141  1.        ]]
      covariance=[[ 0.93780488 -0.13414634]
                  [-0.13414634  0.02439024]]
      dof=3
      eof=5.299876973568286
      iterations=22
      params=
        ResultParameters(
          ResultParameter(name='a1', value=0.5224390243941934, uncert=5.132418149940028, label=None),
          ResultParameter(name='a2', value=4.4068292682920465, uncert=0.8277017245089597, label=None)
        )
    )

Using the *result* object and the :meth:`~msl.nlf.model.Model.evaluate` method,
the residuals can be calculated

.. code-block:: pycon

    >>> y - model.evaluate(x, result)
    array([ 0.22663415,  4.47570732, -7.16      , -0.99570732,  3.45336585])

.. _nlf-save-load:

Save and Load .nlf Files
------------------------
A :class:`~msl.nlf.model.Model` can be saved to a file and loaded from a file.
The file that is created with **msl-nlf** can also be opened in the Delphi
GUI application and a **.nlf** file that is created in the Delphi GUI application
can be loaded in **msl-nlf**. See the :meth:`~.msl.nlf.model.Model.save` method
and the :func:`~msl.nlf.load` function for more details.

.. invisible-code-block: pycon

    >>> import os
    >>> if os.path.isfile('samples.nlf'): os.remove('samples.nlf')

.. code-block:: python

    # Create a model
    from msl.nlf import LinearModel
    model = LinearModel()
    model.fit([1, 2, 3], [0.07, 0.27, 0.33])

    # Save the model to a file.
    # The results of the fit are not written to the file, so if you are
    # opening 'samples.nlf' in the Delphi GUI, click the Calculate button
    # and the Results table and the Graphs will be updated.
    model.save('samples.nlf')


    # At a later date, load the file and perform the fit
    from msl.nlf import load
    loaded = load('samples.nlf')
    results = loaded.fit(loaded.x, loaded.y, params=loaded.params)

.. invisible-code-block: pycon

    >>> os.remove('samples.nlf')

.. _nlf-context-manager:

A Model as a Context Manager
----------------------------
The fit function in the DLL reads the information it needs for the fitting process
from RAM but also from files on the hard disk. Configuration (and perhaps correlation)
files are written to a temporary directory for the DLL function to read from. This
temporary directory should automatically get deleted when you are done using the
:class:`~msl.nlf.model.Model` (when the objects reference count is 0 and gets
garbage collected).

Also, if loading a 32-bit DLL in 64-bit Python (see :ref:`nlf-32vs64`) a client-server
application starts in the background when a :class:`~msl.nlf.model.Model` is
created. Similarly, the client-server application should automatically shut down
when you are done using the :class:`~msl.nlf.model.Model`.

A :class:`~msl.nlf.model.Model` can be used as a context manager (see :ref:`with`)
which will delete the temporary directory (and shut down the client-server
application) once the *with* block is finished, for example,

.. code-block:: python

    from msl.nlf import Model

    x = [1, 2, 3, 4, 5]
    y = [1.1, 4.02, 9.2, 16.2, 25.5]

    with Model('a1*x^2', dll='nlf32') as model:  # temporary files created, client-server protocol starts
        result = model.fit(x, y, params=[1])

    # no longer in the 'with' block
    # temporary files have been deleted
    # the client-server protocol has shut down
    # you must create a new Model if you want to use it again

It is your choice if you want to use a :class:`~msl.nlf.model.Model` as a
context manager. There is no difference in performance, but the *cleanup*
steps are more likely to occur when used as a context manager.
