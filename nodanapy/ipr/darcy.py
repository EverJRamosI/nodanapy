import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties

class Darcy(GasProperties):
    def __init__(self, pressure, temperature, specific_gravity, permeability_g, skin, h, rw, re, dew_pressure: int|float=0, num_pressure: int=25):
        super().__init__(pressure, temperature, specific_gravity)
        self.permeability_g = permeability_g
        self.skin = skin
        self.h = h
        self.rw = rw
        self.re = re
        self.dew_pressure = dew_pressure
        self.num_pressure = num_pressure
        
    def _a_flow_(self):
        return ((1424*self.viscosity_gas()*self.factor_compressibility_gas())/(self.specific_gravity*self.h))*(np.log(0.472*(self.re/self.rw))+self.skin)
    
    def _b_flow_(self):
        beta = (2.33*1e10/(self.permeability_g**1.201))
        return (3.16e-11*beta*self.specific_gravity*self.skin*self.temperature)/((self.h**2)*self.rw)
    
    def flow_gas(self):
        p_diff = np.linspace(self.pressure, 14.7, self.num_pressure)
        return ((self._a_flow_()+np.sqrt((self._a_flow_()**2)+(4*self._b_flow_()*((self.pressure**2)-(p_diff**2)))))/(2*self._b_flow_()))/1000
    
    
ipr1 = Darcy(1309, 600, 0.673, 11, 9.95, 76, 0.3542, 909)

print(ipr1.flow_gas())

