import re
from collections import namedtuple

from .datatypes import FitMethod
from .model import LoadedModel
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
    'load',
    'version_info',
    'FitMethod',
    'InputParameter',
    'InputParameters',
    'Model',
    'ConstantModel',
    'CosineModel',
    'ExponentialModel',
    'GaussianModel',
    'LinearModel',
    'PolynomialModel',
    'SineModel',
)


def load(path: str, *, dll: str = None) -> LoadedModel:
    """Load a **.nlf** file that was created by the Delphi GUI.

    Parameters
    ----------
    path
        The path to a **.nlf** file.
    dll
        Passed to the *dll* keyword argument in :class:`~msl.nlf.model.Model`.

    Returns
    -------
    :class:`~msl.nlf.model.LoadedModel`
        The loaded model.
    """
    # Nonlinear-Fitting/NLF DLL/NLFDLLMaths.pas
    # TFittingMethod=(LM,AmLS,AmMD,AmMM,PwLS,PwMD,PwMM);
    methods = {
        0: Model.FitMethod.LM,
        1: Model.FitMethod.AMOEBA_LS,
        2: Model.FitMethod.AMOEBA_MD,
        3: Model.FitMethod.AMOEBA_MM,
        4: Model.FitMethod.POWELL_LS,
        5: Model.FitMethod.POWELL_MD,
        6: Model.FitMethod.POWELL_MM,
    }

    from .loader import _load
    file = _load(path)

    options = dict(
        correlated=file['correlated_data'],
        delta=file['delta'],
        max_iterations=file['max_iterations'],
        fit_method=methods[file['fitting_method']],
        second_derivs_B=file['second_derivs_B'],
        second_derivs_H=file['second_derivs_H'],
        tolerance=file['tolerance'],
        uy_weights_only=file['uy_weights_only'],
        weighted=file['weighted'],
    )

    loaded = LoadedModel(file['equation'], dll=dll, **options)
    if file['correlated_data']:
        import numpy as np
        for i, j in np.argwhere(file['is_correlated']):
            matrix = file['corr_coeff'][i, j]
            n1 = 'Y' if i == 0 else f'X{i}'
            n2 = 'Y' if j == 0 else f'X{j}'
            loaded.set_correlation(n1, n2, matrix=matrix)

    for i, (a, c) in enumerate(zip(file['a'], file['constant']), start=1):
        loaded.params[f'a{i}'] = a, c

    # the comments text contains information about the fonts and has \\par for paragraphs
    # {\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang5129{\\fonttbl{\\f0\\fnil\\fcharset0 Times New Roman;}}\r\n
    # \\viewkind4\\uc1\\pard\\f0\\fs20 Correlated and \\par\r\nweighted example\\par\r\n}\r\n\x00
    comments = file.get('comments', '')
    found = re.search(r'\\fs20(?P<comments>.+)', comments, flags=re.DOTALL)
    if found:
        comments = found['comments'][:-4]  # ignore trailing }\r\n\x00
        comments = comments.replace('\\par', '')
        comments = comments.replace('\r\n', '\n')
        comments = comments.replace('\\{', '{')
        comments = comments.replace('\\}', '}')
        comments = comments.strip()

    loaded.comments = comments
    loaded.nlf_path = path
    loaded.nlf_version = str(file['version'])
    loaded.ux = file['ux'] if loaded.num_variables > 1 else file['ux'][0]
    loaded.uy = file['uy']
    loaded.x = file['x'] if loaded.num_variables > 1 else file['x'][0]
    loaded.y = file['y']
    return loaded
