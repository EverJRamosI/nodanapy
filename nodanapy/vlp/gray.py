import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties
from _properties.oilProperties import OilProperties
from _properties.waterProperties import WaterProperties

class Gray(GasProperties, OilProperties, WaterProperties):
    
    def __init__(self, pressure, temperature, specific_gravity, api, bubble_pressure, salinity, water_cut, gor, internal_diameter, deep_well, qg_max):
        GasProperties.__init__(self, pressure, temperature, specific_gravity)
        OilProperties.__init__(self, pressure, temperature, specific_gravity, api, bubble_pressure)
        WaterProperties.__init__(self, pressure, temperature, salinity)
        self.water_cut = water_cut
        self.gor = gor
        self.internal_diameter = internal_diameter
        self.deep_well = deep_well
        self.qg_max = qg_max
                
    def _flow_(self, num: float=25):
        q_lm = ((self.gor*(self.qg_max*1.2))/(1-self.water_cut))/1000
        q_liq = np.linspace(1, q_lm, num) #np.array([1, 11, 64, 96, 132, 175, 228, 296, 392, 556, 1002, 1484, 2040, 2597, 3153, 3709])
        q_oil = (1-self.water_cut)*q_liq
        q_water = q_liq-q_oil
        lgr = self.gor/(1-self.water_cut)
        q_gas = (q_liq*1000)/lgr
        return [q_water, q_oil, q_gas, q_liq]
    
    def _fractions_liquid_(self):
        flow_values = self._flow_()
        q_oil = flow_values[1]
        q_water = flow_values[0]
        f_oil = q_oil/(q_oil+q_water)
        f_water = 1-f_oil
        return [f_oil, f_water]
        
    def _velocities_(self):
        flow_values = self._flow_()
        q_gas = flow_values[2]
        v_sg = (q_gas*1000*self.factor_volumetric_gas())/(((np.pi/4)*((self.internal_diameter/12)**2))*86400)
        v_sl = (5.615*(flow_values[1]*self.viscosity_oil()+flow_values[0]*self.viscosity_water()))/((np.pi/4)*((self.internal_diameter/12)**2)*86400)
        v_m = v_sg + v_sl
        return [v_sg, v_sl, v_m]
    
    def _no_slip_holdup_(self):
        lamb_liq = self._velocities_()[1]/self._velocities_()[2]
        lamb_gas = 1-lamb_liq
        return [lamb_liq, lamb_gas]
    
    def _densities_(self):
        rho_liq = self.density_oil()*self._fractions_liquid_()[0]+self.density_water()*self._fractions_liquid_()[1]
        rho_gas = self.density_gas()
        rho_ns = rho_liq*self._no_slip_holdup_()[0]+rho_gas*self._no_slip_holdup_()[1]
        return [rho_liq, rho_gas, rho_ns]
    
    def _sigma_tensions_(self):
        sigma_gw = self.tension_water()
        sigma_go = self.tension_oil()
        sigma_liq = (self._fractions_liquid_()[0]*sigma_go+0.617*self._fractions_liquid_()[1]*sigma_gw)/(self._fractions_liquid_()[0]+0.617*self._fractions_liquid_()[1])
        return [sigma_gw, sigma_go, sigma_liq]
    
    def _viscosities_(self):
        mu_water = self.viscosity_water()
        mu_oil = self.viscosity_oil()
        mu_gas = self.viscosity_gas()
        mu_liq = mu_water*self._fractions_liquid_()[1] + mu_oil*self._fractions_liquid_()[0]
        mu_ns = mu_gas*self._no_slip_holdup_()[1]+mu_liq*self._no_slip_holdup_()[0]
        return [mu_water, mu_oil, mu_gas, mu_liq, mu_ns]
    
    def pressure_drop_elevation(self):
        N1 = 453.592*(((self._densities_()[2]**2)*(self._velocities_()[2]**4))/(32.17*self._sigma_tensions_()[2]*(self._densities_()[0]-self._densities_()[1])))
        N2 = 453.592*((32.17*(self._densities_()[0]-self._densities_()[1])*(self.internal_diameter/12))/(self._sigma_tensions_()[1]))
        R_v = self._velocities_()[1]/self._velocities_()[0]
        N3 = 0.0814*(1-0.0554*np.log(1+((730*R_v)/(R_v+1))))
        G = -2.314*((N1*(1+(205/N2)))**N3)
        holdup = 1-((1-np.exp(G))/(R_v+1))
        rho_holdup = holdup*self._densities_()[0]+self._densities_()[1]*(1-holdup)
        delta_pressure = (32.17*rho_holdup*self.deep_well)/(144*32.17)
        return delta_pressure
    
    def _number_reynolds_(self, rugosity: float=0.0001):
        NRe = (self._densities_()[2]*(self.internal_diameter/12))/(0.000672*self._viscosities_()[4])
        f = (-2*np.log10((1/3.7)*(rugosity/self.internal_diameter))+((6.81/NRe)**0.9))**(-2)
        return [NRe, f]
    
    def pressure_drop_friction(self, rugosity: float=0.0001):
        R_v = self._velocities_()[1]/self._velocities_()[0]
        rugosity_o = (28.5*self._sigma_tensions_()[2])/(453.592*self._densities_()[2]*((self._velocities_()[2])**2))
        rugosity_e = np.where(R_v>=0.007, rugosity_o, rugosity + ((rugosity_o - rugosity) / 0.007))
        friction = self._number_reynolds_(rugosity=rugosity_e)[1]
        delta_pressure = (friction*((self._velocities_()[2])**2)*self._densities_()[2]*self.deep_well)/(144*2*32.17*(self.internal_diameter/12))
        return delta_pressure
        
    def bottom_hole_pressure(self):
        dp_elevation = self.pressure_drop_elevation()
        dp_friction = self.pressure_drop_friction()
        dp_total = dp_elevation + dp_friction
        return self.pressure + dp_total
    
# vlp1 = Gray(1309, (140+460), 0.673, 53.7, 149.7, 8415, 0.88, 48.5981, 2.441, 5098, 7783.199)
# vlp2 = Gray(149.7, (84+460), 0.673, 53.7, 149.7, 8415, 0.88, 48.5981, 2.441, 5098, 7783.199)

# print("flow", vlp1._flow_())
# print("fractions", vlp1._fractions_liquid_())
# print("velocities", vlp1._velocities_())
# print("no holdup", vlp1._no_slip_holdup_())
# print("densities", vlp1._densities_())
# print("sigma",vlp1._sigma_tensions_())
# print("viscosity",vlp1._viscosities_())

# import matplotlib.pyplot as plt

# q1 = vlp1._flow_()
# q2 = vlp2._flow_()

# vlp1.pressure_drop_elevation()
# vlp1.pressure_drop_friction(rugosity=0.00065)
# vlp2.pressure_drop_friction(rugosity=0.00065)

# pt1 = vlp1.bottom_hole_pressure()
# pt2 = vlp2.bottom_hole_pressure()

# print("water", q[0])
# print('\n')
# print("oil", q[1])
# print('\n')

# print("gas1", q1[2])
# print("gas2", q2[2])

# print('\n')
# print("Liq", q[3])

# print("p1", pt1)
# print("p2", pt2)


# plt.plot(q1[2], pt1)
# plt.show()
#fig, ax = plt.subplot()
