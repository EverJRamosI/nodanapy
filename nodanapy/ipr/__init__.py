"""
## Inflow Performance Relationship (IPR)

This archive contains classes to determine the IPR.
"""

from nodanapy.ipr.darcy import Darcy
from nodanapy.ipr.fetkovich import Fetkovich
from nodanapy.ipr.lit import LITPD, LITRD
from nodanapy.ipr.vogel import VogelPD, VogelRD

__all__ = ['Darcy', 'Fetkovich', 'LITPD', 'LITRD', 'VogelPD', 'VogelRD']