"""Descriptor definition, calculation, and result management are separate."""

from .definitions import DescriptorDefinition
from .results import ResultStore

__all__ = ["DescriptorDefinition", "ResultStore"]
