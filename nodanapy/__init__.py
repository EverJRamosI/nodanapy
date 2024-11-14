"""
## NodAnaPy

This packages designed to determine the Nodal Analysis of the gas or oil well.

### Key Features:

- Calculate hold-up liquid
- Calculate inflow performance relationship
- Calculate vertical lift performance

### Modules:

- ipr: Classes to determine inflow performance relationship
- vlp: Classes to determine vertical lift performance
- _properties: Classes to calculate the properties of oil, gas and water
- wellNA: File .py where it is located to determine nodal analysis

### License:

This file is part of My Python Library.

My Python Library is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

My Python Library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with My Python Library. If not, see <http://www.gnu.org/licenses/>.

@author: Ever J. Ramos I.
"""

from ipr.darcy import Darcy
from ipr.fetkovich import Fetkovich
from ipr.lit import LITPD, LITRD
from vlp.beggsBrill import BeggsBrill
from vlp.gray import Gray
from vlp.hagedornBrown import HagedornBrown
from .wellNA import WellNAGasP, WellNAGasR, WellNAOilP, WellNAOilR

__all__ = ['Darcy', 'Fetkovich', 'LITPD', 'LITRD', 'BeggsBrill', 'Gray', 'HagedornBrown',
        'WellNAGasP', 'WellNAGasR', 'WellNAOilP', 'WellNAOilR']
