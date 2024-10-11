import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties
from _properties.oilProperties import OilProperties
from _properties.waterProperties import WaterProperties

class Gray:
    def __init__(self, pressure: int|float, temperature: int|float, specific_gravity: float=0.65, 
                api: int|float=40, bubble_pressure: int|float=0, salinity: int|float=1000, water_cut: float=0.0, 
                go_ratio: int|float=500, internal_diameter: int|float=2.5, rugosity: float=0.0001, well_depth: int|float=500, 
                temperature_node: int|float=600, qg_i: int|float=0.01, qg_n: int|float=10000, amount: int=25):
        
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
        self.qg_i = qg_i
        self.qg_n = qg_n
        self.amount = amount
        
        self.wo_ratio = self.water_cut/(1-self.water_cut)
        self.sg_oil = 141.5/(131.5+self.api)
        self.gl_ratio = (self.go_ratio)/(self.wo_ratio+1)
        #self.lg_ratio = self.go_ratio/(1-self.water_cut)
        self.wg_ratio = (self.water_cut*(1/self.go_ratio))/(1-self.water_cut)
        self.area = np.pi/4*((self.internal_diameter/12)**2)
                
        self.delta_qg = np.linspace(self.qg_i, self.qg_n, self.amount)
        self.delta_depth = np.linspace(0, self.well_depth, self.amount)
        self.delta_t = self._delta_temp_()
        
        self._prop_gas = GasProperties(self.pressure, self.temperature, self.specific_gravity)
        self._prop_oil = OilProperties(self.pressure, self.temperature, self.specific_gravity, self.api, bubble_pressure=self.bubble_pressure)
        self._prop_water = WaterProperties(self.pressure, self.temperature, self.salinity)
    
    def _delta_temp_(self):
        gradient = np.abs((self.temperature-self.temperature_node))/self.well_depth
        return self.temperature + gradient*self.delta_depth
        
    def _flow_(self):
        q_gas = ((self._prop_gas.factor_volumetric_gas()*self.qg_i)/1000)/15387
        q_oil = (self._prop_oil.factor_volumetric_oil()*(self.qg_i/1000)/self.go_ratio)/15387
        q_water = (self._prop_water.factor_volumetric_water()*self.wg_ratio*(self.qg_i/1000))/15387
        q_liq = q_oil + q_water
        return [q_water, q_oil, q_gas, q_liq]

    def _velocities_(self):
        *_, q_gas, q_liq = self._flow_()
        v_sg = ((q_gas*1e6)/(self.area))
        v_sl = ((q_liq)/(self.area))
        v_m = v_sg + v_sl
        return [v_sg, v_sl, v_m]
    
    def _fractions_liquid_(self):
        q_water, q_oil, q_gas, q_liq = self._flow_()
        f_oil = q_oil/q_liq
        f_water = 1-f_oil
        return [f_oil, f_water]        
    
    def _no_slip_holdup_(self):
        v_sg, v_sl, v_m = self._velocities_()
        lamb_liq = v_sl/v_m
        lamb_gas = 1-lamb_liq
        return [lamb_liq, lamb_gas]
    
    def _properties_liquid_(self):
        f_o, f_w = self._fractions_liquid_()
        lamb_liq, lamb_gas = self._no_slip_holdup_()
        rho_liq = self._prop_oil.density_oil()*f_o+self._prop_water.density_water()*f_w
        rho_ns = rho_liq*lamb_liq+self._prop_gas.density_gas()*lamb_gas
        sigma_oil = self._prop_oil.tension_oil()
        if sigma_oil < 1:
            sigma_oil = 1
        sigma_water = self._prop_water.tension_water()
        if sigma_water < 1:
            sigma_water = 1
        sigma_liq = (f_o*sigma_oil+0.617*f_w*sigma_water)/(f_o+0.617*f_w)

        mu_liq = self._prop_water.viscosity_water()*f_w + self._prop_oil.viscosity_oil()*f_o
        mu_ns = self._prop_gas.viscosity_gas()*lamb_gas+mu_liq*lamb_liq
        return [rho_liq, rho_ns, sigma_liq, mu_liq, mu_ns]
    
    def holdup(self):
        rho_liq, rho_ns, sigma_liq, *_ = self._properties_liquid_()
        v_sg, v_sl, v_m = self._velocities_()
        sigma_oil = self._prop_oil.tension_oil()
        if sigma_oil < 1:
            sigma_oil = 1
        N1 = 453.592*(((rho_ns**2)*(v_m**4))/(32.17*sigma_liq*(rho_liq-self._prop_gas.density_gas())))
        N2 = 453.592*((32.17*(rho_liq-self._prop_gas.density_gas())*(self.internal_diameter/12))/(sigma_oil))
        R_v = v_sl/v_sg
        N3 = 0.0814*(1-0.0554*np.log(1+((730*R_v)/(R_v+1))))
        G = -2.314*((N1*(1+(205/N2)))**N3)
        holdup = 1-((1-np.exp(G))/(R_v+1))
        return holdup
    
    def _rho_m_(self):
        rho_liq, *_ = self._properties_liquid_()
        holdup = self.holdup()
        return holdup*rho_liq+self._prop_gas.density_gas()*(1-holdup)
    
    def pressure_drop_elevation(self):
        rho_holdup = self._rho_m_()
        delta_pressure = (32.17*rho_holdup*self.well_depth)/(32.17)
        return delta_pressure
    
    def _number_reynolds_(self):
        *_, q_liq = self._flow_()
        v_sg, v_sl, v_m = self._velocities_()
        rho_liq, rho_ns, sigma_liq, mu_liq, mu_ns = self._properties_liquid_()
        Hl = self.holdup()
        M = self.sg_oil*350.52*(1/(1+self.wo_ratio)) + (self._prop_water.density_water()/62.42)*350.52*(self.wo_ratio/(1+self.wo_ratio)) + self.specific_gravity*0.0764*self.gl_ratio
        R_v = v_sl/v_sg
        NRe = 2.2e-2*((q_liq*15387*M)/((self.internal_diameter/12)*(mu_liq**Hl)*(self._prop_gas.viscosity_gas()**(1-Hl))))
        #NRe = (rho_ns*v_m*(self.internal_diameter/12))/(0.000672*mu_ns)
        rugosity_o = (28.5*sigma_liq)/(453.592*rho_ns*((v_m)**2))
        if R_v >= 0.007:
            rugosity = rugosity_o
        else:
            rugosity = self.rugosity + R_v*((rugosity_o - self.rugosity)/0.007)
        f = (-2*np.log10((1/3.7)*(rugosity/self.internal_diameter))+((6.81/NRe)**0.9))**(-2)
        return [NRe, f]
    
    def pressure_drop_friction(self):
        *_, v_m = self._velocities_()
        rho_liq, rho_ns, *_ = self._properties_liquid_()
        friction = self._number_reynolds_()[1]
        return rho_ns*(friction*((v_m)**2))/(2*32.17*(self.internal_diameter/12))
    
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
        
        for flow_value in self.delta_qg:
            self.qg_i = flow_value
            q_w, q_o, q_g, q_l = self._flow_()
            pwf, *_ = self.pressure_traverse()
            pwfi.append(pwf[-1])
            qoi.append(q_o*15387)
            qwi.append(q_w*15387)
            qgi.append(q_g*86400)
            qli.append(q_l*15387)
            
        return [np.array(qli), np.array(qgi), np.array(qoi), np.array(qwi), np.array(pwfi)]

if __name__ == "__main__":
    #import time
    #time_start = time.time()
    well = Gray(480, (100+460), qg_i=100)# 0.673, 53.7, 149.7, 8415, 0.88, 48.5981, 2.441, 5098, 7800, 5)
    # print(well._properties_())
    #print(well._flow_())
    #print(well._velocities_())
    #print(well.outflow())

    #import matplotlib.pyplot as plt

    #h = well.delta_depth
    #p, dp, hl = well.pressure_traverse()
    #
    #fig, ax = plt.subplots(1, 3)
    #ax[0].invert_yaxis()
    #ax[1].invert_yaxis()
    #ax[2].invert_yaxis()
    #ax[0].plot(dp, h)
    #ax[1].plot(p, h)
    #ax[2].plot(hl, h)
    #plt.show()
    
    #ql, qg, qo, qw, pw = well.outflow()
    #print(ql, qg, qo, qw, pw)
    #plt.plot(qg, pw)
    #time_end = time.time()
    #print('time', time_end-time_start)
    #plt.show()