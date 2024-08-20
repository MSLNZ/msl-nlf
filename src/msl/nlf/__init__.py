import re
from collections import namedtuple

from .datatypes import FitMethod
from .datatypes import ResidualType
from .model import LoadedModel
from .model import Model
from .models import ConstantModel
from .models import ExponentialModel
from .models import GaussianModel
from .models import LinearModel
from .models import PolynomialModel
from .models import SineModel
from .parameter import InputParameter
from .parameter import InputParameters
from .__about__ import __version__, version_tuple

__all__ = (
    'ConstantModel',
    'ExponentialModel',
    'FitMethod',
    'GaussianModel',
    'InputParameter',
    'InputParameters',
    'LinearModel',
    'Model',
    'PolynomialModel',
    'ResidualType',
    'SineModel',
    'load',
    'version_info'
)


def load(path: str, *, dll: str = None) -> LoadedModel:
    """Load a **.nlf** file.

    No information about the fit results are read from the file. The fit
    equation, the fit options and the correlation coefficients have been
    set in the :class:`~msl.nlf.model.LoadedModel` that is returned, but
    you must specify the *x*, *y*, *params*, *ux* and/or *uy* attributes
    of the :class:`~msl.nlf.model.LoadedModel` to the
    :meth:`~msl.nlf.model.Model.fit` method (or specify different data
    to the :meth:`~msl.nlf.model.Model.fit` method).

    Parameters
    ----------
    path
        The path to a **.nlf** file. The file could have been created by the
        Delphi GUI application or by the :meth:`~msl.nlf.model.Model.save` method.
    dll
        Passed to the *dll* keyword argument in :class:`~msl.nlf.model.Model`.

    Returns
    -------
    :class:`~msl.nlf.model.LoadedModel`
        The loaded model.

    Examples
    --------
    .. invisible-code-block: pycon

        >>> from msl.nlf import LinearModel
        >>> m = LinearModel()
        >>> results = m.fit([1, 2, 3], [0.07, 0.27, 0.33])
        >>> m.save('samples.nlf', overwrite=True)

    >>> from msl.nlf import load
    >>> loaded = load('samples.nlf')
    >>> results = loaded.fit(loaded.x, loaded.y, params=loaded.params)

    .. invisible-code-block: pycon

        >>> import os
        >>> if os.path.isfile('samples.nlf'): os.remove('samples.nlf')

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

    # Nonlinear-Fitting/NLF DLL/NLFDLLMaths.pas
    # TResidualType=(dxVx,dyVx,dxVy,dyVy);
    res_types = {
        0: ResidualType.DX_X,
        1: ResidualType.DY_X,
        2: ResidualType.DX_Y,
        3: ResidualType.DY_Y,
    }

    from .loader import _load
    file = _load(path)

    options = dict(
        absolute_residuals=file['absolute_res'],
        correlated=file['correlated_data'],
        delta=file['delta'],
        max_iterations=file['max_iterations'],
        fit_method=methods[file['fitting_method']],
        residual_type=res_types[file['residual_type']],
        second_derivs_B=file['second_derivs_B'],
        second_derivs_H=file['second_derivs_H'],
        tolerance=file['tolerance'],
        uy_weights_only=file['uy_weights_only'],
        weighted=file['weighted'],
    )

    mod = LoadedModel(file['equation'], dll=dll, **options)
    if file['correlated_data']:
        import numpy as np
        for i, j in np.argwhere(file['is_correlated']):
            matrix = file['corr_coeff'][i, j]
            n1 = 'Y' if i == 0 else f'X{i}'
            n2 = 'Y' if j == 0 else f'X{j}'
            mod.set_correlation(n1, n2, matrix=matrix)

    for i, (a, c) in enumerate(zip(file['a'], file['constant']), start=1):
        mod.params[f'a{i}'] = a, c

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

    mod.comments = comments
    mod.nlf_path = path
    mod.nlf_version = str(file['version'])
    mod.ux = file['ux']
    mod.uy = file['uy']
    mod.x = file['x']
    mod.y = file['y']
    return mod
