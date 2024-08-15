import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties

class Darcy: 
    def __init__(self, pressure: int|float, temperature: int|float, specific_gravity: float=0.65, permeability_g: int|float=10, skin: int|float=0, height_formation: int|float=10, well_radius: int|float=0.35, external_radius: int|float=1000):
        self.pressure = pressure
        self.temperature = temperature
        self.specific_gravity = specific_gravity
        self.permeability_g = permeability_g
        self.skin = skin
        self.height_formation = height_formation
        self.well_radius = well_radius
        self.external_radius = external_radius
        
    def _properties_gas_(self):
        properties = GasProperties(self.pressure, self.temperature, self.specific_gravity)
        mu_gas = properties.viscosity_gas()
        z_gas = properties.factor_compressibility_gas()
        return [mu_gas, z_gas]
        
    def _p_diff_(self, range: int=25):
        return np.linspace(self.pressure, 14.7, range)
        
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
    
    def darcy(self, range: int=25):
        qg = self.flow_gas()
        pwf = self._p_diff_(range=range)
        return [qg, pwf]
    
#if __name__ == "__main__":
    
#    ipr1 = Darcy(1309, 600)

#     p = ipr1._p_diff_()
#     q = ipr1.flow_gas()
#    total = ipr1.darcy(10)
#    print(total)
    # import matplotlib.pyplot as plt

    # plt.plot(q, p)
    # plt.title('IPR')
    # plt.ylabel('Pwf(psia)')
    # plt.xlabel('Qg(Mscf/D)')
    # plt.show()

