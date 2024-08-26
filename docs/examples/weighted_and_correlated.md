This example uses a [Model][msl.nlf.model.Model] with the equation specified as a string. The independent variable (stimulus) data is two-dimensional (i.e., contains $x_1$ and $x_2$ variables). There are uncertainties in both $x$ and $y$ variables and the $y-y$ correlation coefficient is 0.5, the $x_1-x_1$ correlation coefficient is 0.8 and the [set_correlation][msl.nlf.model.Model.set_correlation] method is called to set the correlation matrices in the model.

Prepare the model

```python
import numpy as np
from msl.nlf import Model

# Sample data
x = np.array([[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]])
y = np.array([1.1, 1.9, 3.2, 3.7])

# Standard uncertainties in x and y
ux = np.array([[0.01, 0.02, 0.03, 0.04], [0.002, 0.004, 0.006, 0.008]])
uy = np.array([0.5, 0.5, 0.5, 0.5])

# Initial guess
guess = np.array([0, 0.9, 0])

# Specify the equation as a string
model = Model("a1+a2*(x1+exp(a3*x1))+x2")

# Set the options for a weighted and correlated fit
model.options(weighted=True, correlated=True)

# Define the correlation coefficient matrices, the value is set in the
# off-diagonal matrix elements of the correlation matrix
model.set_correlation("y", "y", value=0.5)
model.set_correlation("x1", "x1", value=0.8)
```

To see a summary of the data that would be sent to the fit function in the shared library, call the [fit][msl.nlf.model.Model.fit] method with `debug=True` and print the returned object (an instance of [Input][msl.nlf.datatypes.Input] is returned)

```python
model_input = model.fit(x, y, params=guess, uy=uy, ux=ux, debug=True)
print(model_input)
```

The summary that is printed is

<!-- invisible-code-block: python
import os
with open('docs/assets/example_weighted_correlated_input.txt', mode='wt') as fp:
    out = []
    lines = str(model_input).splitlines()
    for line in lines:
        if 'path=' in line:
            _, path = line.split('=')
            out.append(f"          path='.../{os.path.basename(path)}")
        else:
            out.append(line)
    fp.write('\n'.join(out))
-->

```py
--8<-- "docs/assets/example_weighted_correlated_input.txt"
```

To see a summary of the fit result, call the [fit][msl.nlf.model.Model.fit] method with `debug=False` (which is also the default value) and print the returned object (an instance of [Result][msl.nlf.datatypes.Result] is returned)

```python
result = model.fit(x, y, params=guess, uy=uy, ux=ux, debug=False)
print(result)
```

The summary that is printed is

<!-- invisible-code-block: python
with open('docs/assets/example_weighted_correlated_result.txt', mode='wt') as fp:
    fp.write(str(result))
-->

```py
--8<-- "docs/assets/example_weighted_correlated_result.txt"
```
