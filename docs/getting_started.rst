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

If you want to have control over which parameters should be held constant during the
fitting process and which are allowed to vary or if you want to assign a label to a
parameter, you need to create an :class:`~msl.nlf.parameter.InputParameters` instance.

In this case, we will use one of the built-in models, :class:`~msl.nlf.models.LinearModel`,
to perform the linear fit and create :class:`~msl.nlf.parameter.InputParameters`.
We use the :class:`~msl.nlf.parameter.InputParameters` instance to provide an initial
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

There are multiple ways to add a parameter to an :class:`~msl.nlf.parameter.InputParameters`
instance. Above, we showed that calling :meth:`~msl.nlf.model.Model.create_parameters`
is one way to create an :class:`~msl.nlf.parameter.InputParameters` instance. It can also be
instantiated directly

.. code-block:: pycon

    >>> from msl.nlf import InputParameters
    >>> params = InputParameters()

To add a parameter to the collection of :class:`~msl.nlf.parameter.InputParameters`,
you could explicitly add an instance of an :class:`~msl.nlf.parameter.InputParameter`
object (using the :meth:`~msl.nlf.parameter.InputParameters.add` method or as one would
add items to a :class:`dict` -- using square brackets),

.. code-block:: pycon

    >>> from msl.nlf import InputParameter
    >>> a1 = params.add(InputParameter('a1', 1))
    >>> a2 = params.add(InputParameter('a2', 2, constant=True))
    >>> a3 = params.add(InputParameter('a3', 3, constant=True, label='label-3'))
    >>> params['a4'] = InputParameter('a4', 4)

you could also specify positional arguments (or a :class:`tuple`)

.. code-block:: pycon

    >>> a5 = params.add('a5', 5)
    >>> a6 = params.add('a6', 6, True)
    >>> a7 = params.add('a7', 7, False, 'label-7')
    >>> params['a8'] = 8
    >>> params['a9'] = 9, True
    >>> params['a10'] = 10, True, 'label-10'

or, you could specify keyword arguments (or a :class:`dict`),

.. code-block:: pycon

    >>> a11 = params.add(name='a11', value=11)
    >>> a12 = params.add(name='a12', value=12, constant=True)
    >>> a13 = params.add(name='a13', value=13, label='label-13')
    >>> a14 = params.add(name='a14', value=14, constant=False, label='label-14')
    >>> params['a15'] = {'value': 15}
    >>> params['a16'] = {'value': 16, 'constant': True}
    >>> params['a17'] = {'value': 17, 'label': 'label-17'}
    >>> params['a18'] = {'value': 18, 'constant': False, 'label': 'label-18'}

There is a :meth:`~msl.nlf.parameter.InputParameters.add_many` method as well.

Iterate through the collection of input parameters to see what it contains

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

You can get a specific parameter by *name* or by *label* (provide that the
*label* is not :data:`None`)

.. code-block:: pycon

    >>> params['a3']
    InputParameter(name='a3', value=3.0, constant=True, label='label-3')
    >>> params['label-14']
    InputParameter(name='a14', value=14.0, constant=False, label='label-14')

and you can update a parameter by its *name* or by *label* (the update is
performed *in-place*, i.e., a copy of the parameter is not created)

.. code-block:: pycon

    >>> params.update('a1', value=5.3, label='intercept')
    InputParameter(name='a1', value=5.3, constant=False, label='intercept')
    >>> params.update('label-7', value=1e3, constant=True, label='amplitude')
    InputParameter(name='a7', value=1000.0, constant=True, label='amplitude')

See the :ref:`nlf-examples` for further help.
