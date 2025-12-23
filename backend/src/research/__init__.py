"""
Research Lab - Agent C Implementation

Provides outcome labeling and statistical experimentation tools.
"""

from .labeler import get_outcome
from .experiment_runner import ExperimentRunner

__all__ = ["get_outcome", "ExperimentRunner"]

