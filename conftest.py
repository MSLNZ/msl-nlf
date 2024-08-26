"""Set up testing.

This file must remain at the root directory of the repository so that Sybil
can find the doctests.
"""

import sys
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

import matplotlib as mpl
import pytest
from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser

try:
    import GTC  # type: ignore[import-untyped]
except ImportError:
    GTC = None

mpl.use("svg")

# Sybil does not support implicit namespace packages.
# Add msl to sys.path to avoid getting:
#   ModuleNotFoundError: No module named 'nlf.datatypes'
sys.path.append("src/msl")

# For NIST_datasets to be importable
sys.path.append("tests")


@pytest.fixture()
def no_gtc() -> bool:
    """Checks if GTC is installed."""
    return GTC is None


pytest_collect_file = Sybil(
    parsers=[
        PythonCodeBlockParser(
            future_imports=["annotations"],
            doctest_optionflags=NORMALIZE_WHITESPACE | ELLIPSIS,
        ),
        SkipParser(),
    ],
    patterns=["*.md", "*.py"],
    fixtures=["no_gtc"],
).pytest()
