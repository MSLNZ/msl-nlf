import re
from collections import namedtuple

from .model import Model
from .models import ConstantModel
from .models import CosineModel
from .models import ExponentialModel
from .models import GaussianModel
from .models import LinearModel
from .models import PolynomialModel
from .models import SineModel
from .parameter import InputParameter
from .parameter import InputParameters

__author__ = 'Measurement Standards Laboratory of New Zealand'
__copyright__ = '\xa9 2023, ' + __author__

# The value of the version should follow the DLL version and Python bug fixes
# {DLL major}.{DLL minor}.{Python patch, reset to 0 when DLL major or DLL minor changes}
__version__ = '5.41.0.dev0'

_v = re.search(r'(\d+)\.(\d+)\.(\d+)[.-]?(.*)', __version__).groups()

version_info = namedtuple('version_info', 'major minor micro releaselevel')(int(_v[0]), int(_v[1]), int(_v[2]), _v[3])
""":obj:`~collections.namedtuple`: Contains the version information as a (major, minor, micro, releaselevel) tuple."""

__all__ = (
    'version_info',
    'InputParameter',
    'InputParameters',
    'Model',
    'CosineModel',
    'ExponentialModel',
    'GaussianModel',
    'LinearModel',
    'ConstantModel',
    'PolynomialModel',
    'SineModel',
)
