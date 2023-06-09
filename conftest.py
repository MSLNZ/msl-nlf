"""
Initializes Sybil to test the examples in docstrings and .rst files.

Sets the matplotlib backend to be 'svg' in order to not block the tests when
plt.show() is called within ".. code-block:: python" directives in .rst files.
"""
from doctest import ELLIPSIS
from doctest import NORMALIZE_WHITESPACE

import matplotlib
from sybil import Sybil
from sybil.parsers.rest import DocTestParser
from sybil.parsers.rest import PythonCodeBlockParser

matplotlib.use('svg')

pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=NORMALIZE_WHITESPACE | ELLIPSIS),
        PythonCodeBlockParser(future_imports=['annotations']),
    ],
    patterns=['*.rst', '*.py'],
).pytest()
