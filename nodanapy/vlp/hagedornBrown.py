import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties
from _properties.oilProperties import OilProperties
from _properties.waterProperties import WaterProperties

class HagedornBrown:
    def __init__(self, pressure: int|float, temperature: int|float, specific_gravity: float=0.65, 
                api: int|float=40, bubble_pressure: int|float=0, salinity: int|float=1000, water_cut: float=0.0, 
                go_ratio: int|float=500, internal_diameter: int|float=2.5, rugosity: float=0.0001, well_depth: int|float=5000, 
                temperature_node: int|float=600, qo_i: int|float=0.01, qo_n: int|float=1000, amount: int=25):
        
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
        self.well_depth = well_depth
        self.temperature_node = temperature_node
        self.qo_i = qo_i
        self.qo_n = qo_n
        self.amount = amount
        
        self.wo_ratio = self.water_cut/(1-self.water_cut)
        self.sg_oil = 141.5/(131.5+self.api)
        self.gl_ratio = (self.go_ratio)/(self.wo_ratio+1)
        self.area = np.pi/4*((self.internal_diameter/12)**2)
        
        self.delta_qo = np.linspace(self.qo_i, self.qo_n, self.amount)
        self.delta_depth = np.linspace(0, self.well_depth, self.amount)
        self.delta_t = self._delta_temp_()
        
        self._prop_gas = GasProperties(self.pressure, self.temperature, self.specific_gravity)
        self._prop_oil = OilProperties(self.pressure, self.temperature, self.specific_gravity, self.api, bubble_pressure=self.bubble_pressure)
        self._prop_water = WaterProperties(self.pressure, self.temperature, self.salinity)
    
    def _delta_temp_(self):
        gradient = np.abs((self.temperature-self.temperature_node))/self.well_depth
        return self.temperature + gradient*self.delta_depth
        
    def _properties_liquid_(self):
        self.rho_liq = self._prop_oil.density_oil()*(1/(1+self.wo_ratio)) + self._prop_water.density_water()*(self.wo_ratio/(1+self.wo_ratio))
        self.mu_liq = self._prop_oil.viscosity_oil()*(1/(1+self.wo_ratio)) + self._prop_water.viscosity_water()*(self.wo_ratio/(1+self.wo_ratio))
        sigma_oil = self._prop_oil.tension_oil()
        if sigma_oil < 1:
            sigma_oil = 1
        sigma_water = self._prop_water.tension_water()
        if sigma_water < 1:
            sigma_water = 1
        self.sigma_liq = sigma_oil*(1/(1+self.wo_ratio)) + sigma_water*(self.wo_ratio/(1+self.wo_ratio))
    
    def _flow_(self):
        #q_oil = (self._prop_oil.factor_volumetric_oil()*self.qo_i)/15387
        q_oil = self.qo_i
        #q_water = (self._prop_water.factor_volumetric_water()*self.wo_ratio*self.qo_i)/15387
        q_water = self.wo_ratio*self.qo_i
        q_liq = q_oil+q_water
        #if (self.go_ratio - self.rs_oil) < 0:
        #    q_gas = 0
        #else:
        #    q_gas = self.fvf_gas*(self.go_ratio-self.rs_oil-(self.rs_water*self.wo_ratio))*self.qo_i/86400
        #q_gas = (self._prop_gas.factor_volumetric_gas()*self.qo_i*self.go_ratio)/15387
        q_gas = self.qo_i*self.go_ratio
        return [q_water, q_oil, q_gas, q_liq]
    
    def _velocities_(self):
        qw, qo, qg, ql = self._flow_()
        v_sl = (((self._prop_oil.factor_volumetric_oil()*qo)/15387) + ((self._prop_water.factor_volumetric_water()*qw)/15387))/self.area
        v_sg = ((self._prop_gas.factor_volumetric_gas()*qg)/15387)/self.area
        v_m = v_sl + v_sg
        return [v_sl, v_sg, v_m]
    
    def holdup(self):
        self._properties_liquid_()
        vsl, vsg, vm = self._velocities_()
        
        A = 1.071 - ((0.2218*((vm)**2))/(self.internal_diameter/12))
        
        if A < 0.13:
            A = 0.13
        
        B = vsg/vm
        
        if B > A:
            NLV = 1.938*vsl*((self.rho_liq/self.sigma_liq)**(1/4))
            NGV = 1.938*vsg*((self.rho_liq/self.sigma_liq)**(1/4))
            ND = 120.872*(self.internal_diameter/12)*(np.sqrt(self.rho_liq/self.sigma_liq))
            NL = 0.15726*self.mu_liq*((1/(self.rho_liq*(self.sigma_liq**3)))**(1/4))
            X1 = np.log10(NL) + 3
            Y = -2.69851 + (0.51841*X1) - (0.551*(X1**2)) + (0.54785*(X1**3)) - (0.12195*(X1**4))
            CNL = 10**Y
            
            X2 = (NLV*(self.pressure**0.1)*CNL)/((NGV**0.575)*(14.7**0.1)*ND)
            
            holdup_psi = -0.10307 + 0.61777*(np.log10(X2)+6) - 0.63295*((np.log10(X2)+6)**2) + 0.29598*((np.log10(X2)+6)**3) - 0.0401*((np.log10(X2)+6)**4)
            
            X3 = (NGV*(NL**0.38))/(ND**2.14)
            
            if X3 < 0.01:
                X3 = 0.01
            
            psi = 0.91163 - 4.82176*X3 + 1232.25*(X2**2) - 22253.6*(X3**3) + 116174.3*(X3**4)
            
            holdup_liq = holdup_psi*psi
    
        else:
            vs = 0.8
            holdup_liq = 1 - 0.5*(1 + (vm/vs) - np.sqrt(((1 + (vm/vs))**2) - (4*(vsg/vs))))
        
        if holdup_liq > 1.0:
            holdup_liq = 1.0
        
        return holdup_liq
    
    def _rho_m_(self):
        self._properties_liquid_()
        Hl = self.holdup()
        return self.rho_liq*Hl+self._prop_gas.density_gas()*(1-Hl)
    
    def _number_reynolds_(self):
        qw, qo, *_ = self._flow_()
        q_liq = ((self._prop_oil.factor_volumetric_oil()*qo)/15387) + ((self._prop_water.factor_volumetric_water()*qw)/15387)
        Hl = self.holdup()
        M = self.sg_oil*350.52*(1/(1+self.wo_ratio)) + (self._prop_water.density_water()/62.42)*350.52*(self.wo_ratio/(1+self.wo_ratio)) + self.specific_gravity*0.0764*self.gl_ratio
        NRe = 2.2e-2*((q_liq*15387*M)/((self.internal_diameter/12)*(self.mu_liq**Hl)*(self._prop_gas.viscosity_gas()**(1-Hl))))
        if NRe < 4000:
            f = (-2*np.log10((1/3.7)*(self.rugosity/self.internal_diameter))+((6.81/NRe)**0.9))**(-2)
        else:
            f = 1/(-4*np.log10((self.rugosity/3.7065)-(5.0452/NRe*(np.log10((self.rugosity**(1.1098)/2.8257)+((7.149/NRe)**0.8991))))))**2
        return [NRe, f]
    
    def pressure_drop_friction(self):
        qw, qo, *_ = self._flow_()
        q_liq = ((self._prop_oil.factor_volumetric_oil()*qo)/15387) + ((self._prop_water.factor_volumetric_water()*qw)/15387)
        friction = self._number_reynolds_()[1]
        M = self.sg_oil*350.52*(1/(1+self.wo_ratio)) + (self._prop_water.density_water()/62.42)*350.52*(self.wo_ratio/(1+self.wo_ratio)) + self.specific_gravity*0.0764*self.gl_ratio
        delta_pressure = (friction*(M**2)*((q_liq*15387)**2))/(7.413e10*((self.internal_diameter/12)**5)*self._rho_m_())
        return delta_pressure
    
    def pressure_drop_total(self):
        return (self.pressure_drop_friction() + self._rho_m_())/144    

    def pressure_traverse(self):
        pn = self.pressure
        
        p = [pn]
        hl = [self.holdup()]
        dpt = [self.pressure_drop_total()]

        dz_array = np.diff(self.delta_depth)

        for i, dz in enumerate(dz_array, 1):
            self.temperature = self.delta_t[i]

            pi = dpt[i-1] * dz + p[i-1]
            self.pressure = pi
            self._prop_gas = GasProperties(self.pressure, self.temperature, self.specific_gravity)
            self._prop_oil = OilProperties(self.pressure, self.temperature, self.specific_gravity, self.api, bubble_pressure=self.bubble_pressure)
            self._prop_water = WaterProperties(self.pressure, self.temperature, self.salinity)
            h_n = self.holdup()
            dP_n = self.pressure_drop_total()
            
            p.append(pi)
            hl.append(h_n)
            dpt.append(dP_n)
            
        self.temperature = self.delta_t[0]
        self.pressure = pn
        
        return [np.array(p), np.array(dpt), np.array(hl)]
    
    def outflow(self):
        qoi = []
        qwi = []
        qgi = []
        qli = []
        pwfi = []
        
        for flow_value in self.delta_qo:
            self.qo_i = flow_value
            q_w, q_o, q_g, q_l = self._flow_()
            pwf, *_ = self.pressure_traverse()
            pwfi.append(pwf[-1])
            qoi.append(q_o)
            qwi.append(q_w)
            qgi.append(q_g/1000)
            qli.append(q_l)
            
        return [np.array(qli), np.array(qgi), np.array(qoi), np.array(qwi), np.array(pwfi)]

    
if __name__ == "__main__":
    import time
    time_start = time.time()    
    well = HagedornBrown(149.7, (84+460), bubble_pressure=1000, well_depth=8000, qo_i=5.3163, qo_n=85.0577)
    
    #print(well.pressure_traverse_new())
    #print(well.outflow())
    #print(well._velocities_())
    import matplotlib.pyplot as plt
    
    ql, qg, qo, qw, pw = well.outflow()
    print(ql, qg, qo, qw, pw)
    time_end = time.time()
    print('Time', time_end - time_start)
    plt.plot(qg, pw)
    plt.show()
    # h = well.delta_depth
    # p, dp, hl = well.pressure_traverse()
    # print(h, p, dp, hl)
    
    # fig, ax = plt.subplots(1, 3)
    # ax[0].invert_yaxis()
    # ax[1].invert_yaxis()
    # ax[2].invert_yaxis()
    # ax[0].plot(dp, h)
    # ax[1].plot(p, h)
    # ax[2].plot(hl, h)
    #plt.show()
    
    