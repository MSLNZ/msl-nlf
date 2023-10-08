"""
Initializes Sybil to test the examples in docstrings and .rst files.

Sets the matplotlib backend to be 'svg' in order to not block the tests when
plt.show() is called within ".. code-block:: python" directives in .rst files.
"""
from doctest import ELLIPSIS
from doctest import NORMALIZE_WHITESPACE

import matplotlib
import pytest
from sybil import Sybil
from sybil.parsers.rest import DocTestParser
from sybil.parsers.rest import PythonCodeBlockParser
from sybil.parsers.rest import SkipParser

try:
    import GTC
except ImportError:
    GTC = None

matplotlib.use('svg')


@pytest.fixture
def no_gtc():
    return GTC is None


pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=NORMALIZE_WHITESPACE | ELLIPSIS),
        PythonCodeBlockParser(future_imports=['annotations']),
        SkipParser(),
    ],
    patterns=['*.rst', '*.py'],
    fixtures=['no_gtc'],
).pytest()
