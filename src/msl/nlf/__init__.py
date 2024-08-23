"""Wrapper around the Delphi non-linear fitting software."""

from __future__ import annotations

from .__about__ import __version__, version_tuple
from .datatypes import FitMethod, ResidualType
from .loader import load
from .model import LoadedModel, Model
from .models import ConstantModel, ExponentialModel, GaussianModel, LinearModel, PolynomialModel, SineModel
from .parameter import InputParameter, InputParameters
