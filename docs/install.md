# Install
`msl-nlf` is available for installation via the [Python Package Index](https://pypi.org/){:target="_blank"} and may be installed with [pip](https://pip.pypa.io/en/stable/){:target="_blank"}

```console
pip install msl-nlf
```

## Dependencies
* Python 3.8+
* [numpy](https://pypi.org/project/numpy/){:target="_blank"}
* [msl-loadlib](https://pypi.org/project/msl-loadlib/){:target="_blank"} (Windows only)

### Optional Dependencies
The GUM Tree Calculator, [GTC]{:target="_blank"}, is not automatically installed when `msl-nlf` is installed, but it is required to create a correlated ensemble of [uncertain real numbers][uncertain_real_number]{:target="_blank"} from a [Result][msl.nlf.datatypes.Result].

To automatically include [GTC]{:target="_blank"} when installing `msl-nlf` you may use

```console
pip install msl-nlf[gtc]
```

[GTC]: https://pypi.org/project/GTC/
