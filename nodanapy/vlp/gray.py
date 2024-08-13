import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties
from _properties.oilProperties import OilProperties
from _properties.waterProperties import WaterProperties

class Gray(GasProperties, OilProperties, WaterProperties):
    
    def __init__(self, pressure, temperature, specific_gravity, api, bubble_pressure, salinity, water_cut, gor, id, deep_well):
        GasProperties.__init__(self, pressure, temperature, specific_gravity)
        OilProperties.__init__(self, pressure, temperature, specific_gravity, api, bubble_pressure)
        WaterProperties.__init__(self, pressure, temperature, salinity)
        self.water_cut = water_cut
        self.gor = gor
        self.id = id
        self.deep_well = deep_well
                
    def _flow_(self):
        q_liq = 11
        q_oil = (1-self.water_cut)*q_liq
        q_water = q_liq-q_oil
        lgr = self.gor/(1-self.water_cut)
        q_gas = (q_liq*1000)/lgr
        return [q_water, q_oil, q_gas]
    
    def _factors_(self):
        q_oil = self._flow_()[1]
        q_liq = self._flow_()[0]
        f_oil = q_oil/(q_oil+q_liq)
        f_water = 1-f_oil
        return [f_oil, f_water]
        
    def _velocities_(self):
        q_gas = self._flow_()[2]
        v_sg = (q_gas*1000*self.factor_compressibility_gas())/((np.pi/4)*((self.id/12)**2)*86400)
        v_sl = (5.615*(self._flow_()[1]*self.viscosity_oil()+self._flow_()[0]*self.viscosity_water()))/((np.pi/4)*((self.id/12)**2)*86400)
        v_m = v_sg + v_sl
        return [v_sg, v_sl, v_m]
    
    def _no_slip_holdup_(self):
        lamb_liq = self._velocities_()[1]/self._velocities_()[2]
        lamb_gas = 1-lamb_liq
        return [lamb_liq, lamb_gas]
    
    def _densities_(self):
        rho_liq = self.viscosity_gas()*self._factors_()[0]+self.viscosity_water()*self._factors_()[1]
        rho_gas = self.density_gas()
        rho_ns = rho_liq*self._no_slip_holdup_()[0]+rho_gas*self._no_slip_holdup_()[1]
        return [rho_liq, rho_gas, rho_ns]
    
    def _sigma_tensions_(self):
        sigma_gw = self.tension_water()
        sigma_go = self.tension_oil()
        sigma_liq = (self._factors_()[0]*sigma_go+0.617*self._factors_()[1]*sigma_gw)/(self._factors_()[0]+0.617*self._factors_()[1])
        return [sigma_gw, sigma_go, sigma_liq]
    
    def _viscosities_(self):
        mu_water = self.viscosity_water()
        mu_oil = self.viscosity_oil()
        mu_gas = self.viscosity_gas()
        mu_liq = mu_water*self._factors_()[1] + mu_oil*self._factors_()[0]
        mu_ns = mu_gas*self._no_slip_holdup_()[1]+mu_liq*self._no_slip_holdup_()[0]
        return mu_ns
    
    def pressure_drop_elevation(self):
        N1 = 453.592*(((self._densities_()[2]**2)*(self._velocities_()[2]**4))/(32.17*self._sigma_tensions_()[2]*(self._densities_()[0]-self._densities_()[1])))
        N2 = 453.592*((32.17*(self._densities_()[0]-self._densities_()[1])*(self.id/12))/(self._sigma_tensions_()[1]))
        R_v = self._velocities_()[1]/self._velocities_()[0]
        N3 = 0.0814*(1-0.0554*np.log(1+((730*R_v)/(R_v+1))))
        G = -2.314*((N1*(1+(205/N2)))**N3)
        holdup = 1-((1-np.exp(G))/(R_v+1))
        rho_holdup = holdup*self._densities_()[0]+self._densities_()[1]*(1-holdup)
        delta_pressure = (32.17*rho_holdup*self.deep_well)/(144*32.17)
        #print(N1, N2, R_v, N3, G, holdup, rho_holdup)
        return delta_pressure
    
# vlp1 = Gray(149.7, (84+460), 0.673, 53.7, 149.7, 8415, 0.88, 48.5981, 2.441, 5098)

# print("velocities", vlp1._velocities_())
# print("no holdup", vlp1._no_slip_holdup_())
# print("densities", vlp1._densities_())
# print("sigma",vlp1._sigma_tensions_())
# print(vlp1.pressure_drop_elevation())