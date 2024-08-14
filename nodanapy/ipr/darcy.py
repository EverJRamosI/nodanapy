import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties

class Darcy(GasProperties):
    def __init__(self, pressure, temperature, specific_gravity, permeability_g, skin, h, rw, re, num_pressure: int=25):
        super().__init__(pressure, temperature, specific_gravity)
        self.permeability_g = permeability_g
        self.skin = skin
        self.h = h
        self.rw = rw
        self.re = re
        self.num_pressure = num_pressure
        
    def _p_diff_(self):
        return np.linspace(self.pressure, 14.7, self.num_pressure)
        
    def _a_flow_(self, p):
        original_pressure = self.pressure
        self.pressure = p
        A = ((1424*self.viscosity_gas()*self.factor_compressibility_gas()*self.temperature)/(self.permeability_g*self.h))*(np.log(0.472*(self.re/self.rw))+self.skin)
        self.pressure = original_pressure
        return A
        
    def _b_flow_(self, p):
        original_pressure = self.pressure
        self.pressure = p
        beta = (2.33*1e10/(self.permeability_g**1.201))
        B = (3.16e-12*beta*self.specific_gravity*self.factor_compressibility_gas()*self.temperature)/((self.h**2)*self.rw)
        self.pressure = original_pressure
        return B
        
    def flow_gas(self):
        p_diff = self._p_diff_()
        flow_results = []
        
        for p in p_diff:
            A = self._a_flow_(p)
            B = self._b_flow_(p)
            flow = ((-A+np.sqrt((A**2)+(4*B*((self.pressure**2)-(p**2)))))/(2*B))
            flow_results.append(flow)
        
        return np.array(flow_results)
    
    
# ipr1 = Darcy(1309, 600, 0.673, 11, 9.95, 76, 0.3542, 909)

# p = ipr1._p_diff_()
# q = ipr1.flow_gas()

# import matplotlib.pyplot as plt

# plt.plot(q, p)
# plt.title('IPR')
# plt.ylabel('Pwf(psia)')
# plt.xlabel('Qg(Mscf/D)')
# plt.show()



