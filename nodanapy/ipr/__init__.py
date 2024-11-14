"""
## Inflow Performance Relationship (IPR)

This archive contains classes to determine the IPR
"""

from .darcy import Darcy
from .fetkovich import Fetkovich
from .lit import LITPD, LITRD
from .vogel import VogelPD, VogelRD

__all__ = ['Darcy', 'Fetkovich', 'LITPD', 'LITRD', 'VogelPD', 'VogelRD']