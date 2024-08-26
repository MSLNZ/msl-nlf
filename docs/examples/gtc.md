This example requires [GTC](https://gtc.readthedocs.io/en/stable/){:target="_blank"} to be installed. If it is not already installed, you may run

```console
pip install GTC
```

This example uses the calibration curve that is defined in Appendix H3 of the GUM[^1]

$$
b(t) = y_1 + y_2 (t-t_0)
$$

where the reference temperature, $t_0$, is chosen to be 20 $^\circ \mathrm{C}$.

This calibration curve translates to the following equation that is passed to a [Model][msl.nlf.model.Model]

$$
f(x; a) = a_1 + a_2 (x-20)
$$

The *intercept* ($a_1$) and *slope* ($a_2$) result parameters are converted to a correlated ensemble of [uncertain real numbers][uncertain_real_number]{:target="_blank"} (via the [to_ureal][msl.nlf.datatypes.Result.to_ureal] method) which are used to calculate the response at a chosen stimulus.

<!-- skip: start if(no_gtc, reason="GTC cannot be imported") -->

```python
from msl.nlf import Model

# Thermometer readings (degrees C)
x = (21.521, 22.012, 22.512, 23.003, 23.507,
     23.999, 24.513, 25.002, 25.503, 26.010, 26.511)

# Observed differences with calibration standard (degrees C)
y = (-0.171, -0.169, -0.166, -0.159, -0.164,
     -0.165, -0.156, -0.157, -0.159, -0.161, -0.160)

# Arbitrary offset temperature (degrees C)
t0 = 20

# Create the model
model = Model(f"a1+a2*(x-{t0})")

# Create an initial guess. Allow the intercept and slope to vary during
# the fitting process and assign helpful labels
params = model.create_parameters([
    ("a1", 1, False, "intercept"),
    ("a2", 1, False, "slope")
])

# Apply the fit
result = model.fit(x, y, params=params)

# Convert the result to a correlated ensemble of uncertain real numbers
intercept, slope = result.to_ureal()
```

The *intercept* and *slope* can be used to calculate a correction for a reading of 30 $^\circ \mathrm{C}$

```python
>>> intercept + slope*(30 - t0)
ureal(-0.14937681268874...,0.004138595752854...,9.0)
```

[^1]: BIPM and IEC and IFCC and ISO and IUPAC and IUPAP and OIML, [*Evaluation of measurement data - Guide to the expression of uncertainty in measurement JCGM 100:2008 (GUM 1995 with minor corrections)*](http://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e.html?csnumber=50461>){:target="_blank"}.
