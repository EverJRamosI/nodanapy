import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties

class Darcy: 
    def __init__(self, pressure: int|float, temperature: int|float, specific_gravity: float=0.65, permeability_g: int|float=10, skin: int|float=0, height_formation: int|float=10, well_radius: int|float=0.35, external_radius: int|float=1000, water_cut: float=0.1, gor: int|float=50, amount: int=25):
        self.pressure = pressure
        self.temperature = temperature
        self.specific_gravity = specific_gravity
        self.permeability_g = permeability_g
        self.skin = skin
        self.height_formation = height_formation
        self.well_radius = well_radius
        self.external_radius = external_radius
        self.water_cut = water_cut
        self.gor = gor
        self.amount = amount
        
    def _properties_gas_(self):
        properties = GasProperties(self.pressure, self.temperature, self.specific_gravity)
        mu_gas = properties.viscosity_gas()
        z_gas = properties.factor_compressibility_gas()
        return [mu_gas, z_gas]
        
    def _p_diff_(self):
        return np.linspace(self.pressure, 14.7, self.amount)
        
    def _a_flow_(self):
        mu, z = self._properties_gas_()
        A = ((1424*mu*z*self.temperature)/(self.permeability_g*self.height_formation))*(np.log(0.472*(self.external_radius/self.well_radius))+self.skin)
        return A
        
    def _b_flow_(self): 
        z = self._properties_gas_()[1]        
        beta = (2.33*1e10/(self.permeability_g**1.201))
        B = (3.16e-12*beta*self.specific_gravity*z*self.temperature)/((self.height_formation**2)*self.well_radius)
        return B
        
    def flow_gas(self):
        p_diff = self._p_diff_()
        flow_results = []
        A = self._a_flow_()
        B = self._b_flow_()
        
        for p in p_diff:
            flow = ((-A+np.sqrt((A**2)+(4*B*((self.pressure**2)-(p**2)))))/(2*B))
            flow_results.append(flow)
        
        return np.array(flow_results)
    
    def darcy(self):
        qg = self.flow_gas()
        ql = ((self.gor*qg)/(1-self.water_cut))/1000
        qo = (1-self.water_cut)*ql
        qw = ql-qo
        pwf = self._p_diff_()
        return [ql, qg, qo, qw, pwf]
    
# if __name__ == "__main__":
    
#     ipr1 = Darcy(1309, 600, 0.673, 11, 9.95, 76, 0.3542, 909, 0.88, 48.6)

#     p = ipr1._p_diff_()
#     q = ipr1.flow_gas()
#     total = ipr1.darcy()
#     print(total)
    # import matplotlib.pyplot as plt

    # plt.plot(q, p)
    # plt.title('IPR')
    # plt.ylabel('Pwf(psia)')
    # plt.xlabel('Qg(Mscf/D)')
    # plt.show()

