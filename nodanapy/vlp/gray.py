import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties
from _properties.oilProperties import OilProperties
from _properties.waterProperties import WaterProperties

class Gray:
    def __init__(self, pressure: int|float, temperature: int|float, specific_gravity: float=0.65, 
                api: int|float=40, bubble_pressure: int|float=0, salinity: int|float=1000, water_cut: float=0.1, 
                go_ratio: int|float=50, internal_diameter: int|float=2.5, rugosity: float=0.0001, deep_well: int|float=500, 
                qg_max: int|float=10000, amount: int=25):
        
        self.pressure = pressure
        self.temperature = temperature
        self.specific_gravity = specific_gravity
        self.api = api
        self.bubble_pressure = bubble_pressure
        self.salinity = salinity
        self.water_cut = water_cut
        self.go_ratio = go_ratio
        self.internal_diameter = internal_diameter
        self.rugosity = rugosity
        self.deep_well = deep_well
        self.qg_max = qg_max
        self.amount = amount
                
    def _properties_gas_(self):
        properties = GasProperties(self.pressure, self.temperature, self.specific_gravity)
        mu_gas = properties.viscosity_gas()
        b_gas = properties.factor_volumetric_gas()
        rho_gas = properties.density_gas()
        return [mu_gas, rho_gas, b_gas]
    
    def _properties_oil_(self):
        properties = OilProperties(self.pressure, self.temperature, self.specific_gravity, self.api, self.bubble_pressure)
        mu_oil = properties.viscosity_oil()
        rho_oil = properties.density_oil()
        sigma_oil = properties.tension_oil()
        return [mu_oil, rho_oil, sigma_oil]
    
    def _properties_water_(self):
        properties = WaterProperties(self.pressure, self.temperature, self.salinity)
        mu_water = properties.viscosity_water()
        rho_water = properties.density_water()
        sigma_water = properties.tension_water()
        return [mu_water, rho_water, sigma_water]
    
    def _flow_(self):
        q_lm = ((self.go_ratio*(self.qg_max*1.2))/(1-self.water_cut))/1000
        q_liq = np.linspace(1, q_lm, self.amount)
        q_oil = (1-self.water_cut)*q_liq
        q_water = q_liq-q_oil
        lgr = self.go_ratio/(1-self.water_cut)
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
        b_g = self._properties_gas_()[2]
        mu_o = self._properties_oil_()[0]
        mu_w = self._properties_water_()[0]
        flow_values = self._flow_()
        q_gas = flow_values[2]
        v_sg = (q_gas*1000*b_g)/(((np.pi/4)*((self.internal_diameter/12)**2))*86400)
        v_sl = (5.615*(flow_values[1]*mu_o+flow_values[0]*mu_w))/((np.pi/4)*((self.internal_diameter/12)**2)*86400)
        v_m = v_sg + v_sl
        return [v_sg, v_sl, v_m]
    
    def _no_slip_holdup_(self):
        lamb_liq = self._velocities_()[1]/self._velocities_()[2]
        lamb_gas = 1-lamb_liq
        return [lamb_liq, lamb_gas]
    
    def _densities_(self):
        rho_o = self._properties_oil_()[1]
        rho_w = self._properties_water_()[1]
        rho_g = self._properties_gas_()[1]
        rho_liq = rho_o*self._fractions_liquid_()[0]+rho_w*self._fractions_liquid_()[1]
        rho_ns = rho_liq*self._no_slip_holdup_()[0]+rho_g*self._no_slip_holdup_()[1]
        return [rho_liq, rho_g, rho_ns]
    
    def _sigma_tensions_(self):
        sigma_gw = self._properties_water_()[2]
        sigma_go = self._properties_oil_()[2]
        sigma_liq = (self._fractions_liquid_()[0]*sigma_go+0.617*self._fractions_liquid_()[1]*sigma_gw)/(self._fractions_liquid_()[0]+0.617*self._fractions_liquid_()[1])
        return [sigma_gw, sigma_go, sigma_liq]
    
    def _viscosities_(self):
        mu_water = self._properties_water_()[0]
        mu_oil = self._properties_oil_()[0]
        mu_gas = self._properties_gas_()[0]
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
    
    def _number_reynolds_(self):
        NRe = (self._densities_()[2]*(self.internal_diameter/12))/(0.000672*self._viscosities_()[4])
        f = (-2*np.log10((1/3.7)*(self.rugosity/self.internal_diameter))+((6.81/NRe)**0.9))**(-2)
        return [NRe, f]
    
    def pressure_drop_friction(self):
        R_v = self._velocities_()[1]/self._velocities_()[0]
        rugosity_o = (28.5*self._sigma_tensions_()[2])/(453.592*self._densities_()[2]*((self._velocities_()[2])**2))
        rugosity_e = np.where(R_v>=0.007, rugosity_o, self.rugosity + ((rugosity_o - self.rugosity) / 0.007))
        friction = self._number_reynolds_()[1]
        delta_pressure = (friction*((self._velocities_()[2])**2)*self._densities_()[2]*self.deep_well)/(144*2*32.17*(self.internal_diameter/12))
        return delta_pressure
        
    def bottom_hole_pressure(self):
        dp_elevation = self.pressure_drop_elevation()
        dp_friction = self.pressure_drop_friction()
        dp_total = dp_elevation + dp_friction
        pwf = self.pressure + dp_total
        return [dp_total, pwf]
    
    def outflow(self):
        qw, qo, qg, ql = self._flow_()
        pwf = self.bottom_hole_pressure()[1]
        return [ql, qg, qo, qw, pwf]


#np.array([1, 11, 64, 96, 132, 175, 228, 296, 392, 556, 1002, 1484, 2040, 2597, 3153, 3709])
if __name__ == "__main__":
    
    vlp1 = Gray(130, (80+460), 0.673, 53.7, 149.7, 8415, 0.88, 48.5981, 2.441, 5098, 7800, 5)
    #
    #q1 = vlp1._flow_()
    #
    #vel = vlp1._velocities_()
    #print("veloc", vel)
    #print("\n")
    #fract = vlp1._fractions_liquid_()
    #print("fraction", fract)
    #print("\n")
    #p1 = vlp1.pressure_drop_elevation()
    #print("elevation", p1)
    #print("\n")
    #p2 = vlp1.pressure_drop_friction(rugosity=0.00065)
    #print("frict", p2)
    #print("\n")
    #pt1 = vlp1.bottom_hole_pressure()
    #print(pt1)
    #print("\n")
    #flow = vlp1._flow_()
    #print("flow", flow, len(flow))
    #print("\n")
    total = vlp1.outflow()
    print("total", total)
#
#   # import matplotlib.pyplot as plt

    #plt.plot(q1[2], pt1)
#    plt.plot(total[1], total[-1])
#    plt.show()
    #fig, ax = plt.subplot()
