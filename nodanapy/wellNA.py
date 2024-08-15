import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ipr.darcy import Darcy
from vlp.gray import Gray

class WellNA:
    def __init__(self, wellhead_pressure, wellhead_temperature, reservoir_pressure, reservoir_temperature, specific_gravity, permeability_g, skin, h, rw, re, num_pressure, api, bubble_pressure, salinity, water_cut, gor, internal_diameter, deep_well):
        #Darcy.__init__(self, pressure, temperature, specific_gravity, permeability_g, skin, h, rw, re, num_pressure)
        #Gray.__init__(self, pressure, temperature, specific_gravity, api, bubble_pressure, salinity, water_cut, gor, internal_diameter, deep_well, qg_max)
        self.specific_gravity = specific_gravity
        self.permeability_g = permeability_g
        self.skin = skin
        self.h = h
        self.rw = rw
        self.re = re
        self.num_pressure = num_pressure
        self.api = api
        self.bubble_pressure = bubble_pressure
        self.salinity = salinity
        self.water_cut = water_cut
        self.gor = gor
        self.internal_diameter = internal_diameter
        self.deep_well = deep_well
        self.wellhead_pressure = wellhead_pressure
        self.wellhead_temperature = wellhead_temperature
        self.reservoir_pressure = reservoir_pressure
        self.reservoir_temperature = reservoir_temperature
        
    def _ipr_(self, num: int=25):
        ipr = Darcy(self.reservoir_pressure, self.reservoir_temperature, self.specific_gravity, self.permeability_g, self.skin, self.h, self.rw, self.re, num_pressure=num)
        pwf = ipr._p_diff_()
        qg = ipr.flow_gas()
        return [pwf, qg]
    
    def _vlp_(self, num: int=25):
        aof = self._ipr_()[1][-1]
        vlp = Gray(self.wellhead_pressure, self.wellhead_temperature, self.specific_gravity, self.api, self.bubble_pressure, self.salinity, self.water_cut, self.gor, self.internal_diameter, self.deep_well, qg_max=aof)
        pwf = vlp.bottom_hole_pressure()
        qg = vlp._flow_(num=num)
        return [pwf, qg[2]]


# well1 = WellNA(149.7, (84+460), 1309, (140+460), 0.673, 11, 9.95, 76, 0.3542, 909, 25, 53.7, 0, 8415, 0.88, 48.5981, 2.441, 5098)

# print("IPR")
# print("Pwf", well1._ipr_()[0], "Qg", well1._ipr_()[1])
# print("VLP")
# print("Pwf", well1._vlp_()[0], "Qg", well1._vlp_()[1])

# ipr = well1._ipr_()
# vlp = well1._vlp_()

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()

# ax.plot(ipr[1], ipr[0])
# ax.plot(vlp[1], vlp[0])

# plt.show()