import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.oilProperties import OilProperties

class Vogel:
    def __init__(self, pressure: int|float, temperature: int|float, bubble_pressure: int|float=0, specific_gravity: float=0.65, api: int|float=40,
                permeability: int|float=10, compressibility: float=0, skin: int|float=0, height_formation: int|float=10, 
                well_radius: int|float=0.35, reservoir_radius: int|float=1000, water_cut: float=0.0, 
                go_ratio: int|float=50, amount: int=25):
        
        self.pressure = pressure
        self.temperature = temperature
        self.bubble_pressure = bubble_pressure
        self.specific_gravity = specific_gravity
        self.api = api
        self.permeability = permeability
        self.compressibility = compressibility
        self.skin = skin
        self.height_formation = height_formation
        self.well_radius = well_radius
        self.reservoir_radius = reservoir_radius
        self.water_cut = water_cut
        self.go_ratio = go_ratio
        self.amount = amount
        
    def _properties_oil_(self):
        properties = OilProperties(self.pressure, self.temperature, self.specific_gravity, self.api, self.bubble_pressure)
        fvf_oil = properties.factor_volumetric_oil(self.compressibility)
        mu_oil = properties.viscosity_oil()
        return [fvf_oil, mu_oil, ]
    
    def productivity_index(self, q_test: int|float=None, pwf_test: int|float=None, flow_regime: str='pc'):
        self.q_test = q_test
        self.pwf_test = pwf_test
        self.flow_regime = flow_regime
        
        bo, mu = self._properties_oil_()
        
        if self.skin < 0:
            if self.q_test is not None and self.pwf_test is not None:
                if self.pressure >= self.bubble_pressure:
                    j = (self.q_test)/(self.pressure-self.pwf_test)
                else:
                    j = (self.q_test)/((self.pressure-self.bubble_pressure)+(self.bubble_pressure/1.8)*(1-0.2*(self.pwf_test/self.bubble_pressure)-0.8*((self.pwf_test/self.bubble_pressure)**2)))
            else:
                if flow_regime.lower() == 'pc':
                    j = (self.permeability*self.height_formation)/(141.2*bo*mu*(np.log(self.reservoir_radius/self.well_radius)-(3/4)+self.skin))
                elif flow_regime.lower() == 'c':
                    j = (self.permeability*self.height_formation)/(141.2*bo*mu*(np.log(self.reservoir_radius/self.well_radius)+self.skin))
                
        elif self.skin > 0:
            if self.q_test is not None and self.pwf_test is not None:
                if self.pressure > self.bubble_pressure:
                    j = (self.q_test)/(self.pressure-self.pwf_test)
                elif self.pressure <= self.bubble_pressure:
                    j = (self.q_test)/((self.pressure-self.bubble_pressure)+(self.bubble_pressure/1.8)*(1-0.2*(self.pwf_test/self.bubble_pressure)-0.8*((self.pwf_test/self.bubble_pressure)**2)))
            else:
                if self.flow_regime.lower()=='pc':
                    j = (self.permeability*self.height_formation)/(141.2*bo*mu*(np.log(self.reservoir_radius/self.well_radius)-(3/4)+self.skin))
                elif self.flow_regime.lower()=='c':
                    j = (self.permeability*self.height_formation)/(141.2*bo*mu*(np.log(self.reservoir_radius/self.well_radius)+self.skin))
        
        else:
            if self.q_test is not None and self.pwf_test is not None:
                if self.pressure > self.bubble_pressure:
                    j = (self.q_test)/(self.pressure-self.pwf_test)
                elif self.pressure <= self.bubble_pressure:
                    j = (self.q_test)/((self.pressure-self.bubble_pressure)+(self.bubble_pressure/1.8)*(1-0.2*(self.pwf_test/self.bubble_pressure)-0.8*((self.pwf_test/self.bubble_pressure)**2)))
            else:
                if self.flow_regime.lower()=='pc':
                    j = (self.permeability*self.height_formation)/(141.2*bo*mu*(np.log(self.reservoir_radius/self.well_radius)-(3/4)+self.skin))
                elif self.flow_regime.lower()=='c':
                    j = (self.permeability*self.height_formation)/(141.2*bo*mu*(np.log(self.reservoir_radius/self.well_radius)+self.skin))
                j = True
        
        return j
    
    def _flow_bif_(self, ):
        return (self.productivity_index(self.q_test, self.pwf_test, self.flow_regime)*(self.pressure-self.bubble_pressure))
    
    def flow_max(self, q_test: int|float=None, pwf_test: int|float=None):
        return self._flow_bif_()+(self.productivity_index(self.q_test, self.pwf_test, self.flow_regime)*self.pressure)/1.8
    
    def _p_diff_(self):
        return np.linspace(self.pressure, 14.7, self.amount)
    
    def inflow(self):
        pwf = self._p_diff_()
        qo = self.flow_max()*(1-0.2*(pwf/self.pressure)-0.8*((pwf/self.pressure)**2))
        qg = qo*self.go_ratio
        ql = ((self.go_ratio*qg)/(1-self.water_cut))/1000
        qw = ql-qo
        return [ql, qg, qo, qw, pwf]
    

# if __name__ == "__main__":
    
#     well1 = Vogel(5651, 590, permeability=8.2, height_formation=53, reservoir_radius=2980, go_ratio=85, skin=5)
    
#     print(well1.productivity_index())
#     print(well1._flow_bif_())
#     print(well1.flow_max())
#     print(well1.inflow())
    
    # qo = well1.inflow()[2]
    # p = well1.inflow()[-1]
    
    # import matplotlib.pyplot as plt
    
    # plt.plot(qo, p)
    # plt.show()