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
                go_ratio: int|float=300, internal_diameter: int|float=2.5, rugosity: float=0.0001, well_depth: int|float=5000, 
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
        
        self.mu_gas = None
        self.fvf_gas = None
        self.rho_gas = None
        self.z_gas = None
        
        self.mu_oil = None
        self.rho_oil = None
        self.sigma_oil = None
        self.rs_oil = None
        self.fvf_oil = None
        
        self.mu_water = None
        self.rho_water = None
        self.sigma_water = None
        self.rs_water = None
        self.fvf_water = None
        
        self.rho_liq = None
        self.mu_liq = None
        self.sigma_liq = None
        
    
    def _delta_temp_(self):
        gradient = np.abs((self.temperature-self.temperature_node))/self.well_depth
        return self.temperature + gradient*self.delta_depth
    
    def _properties_(self):
        properties_gas = GasProperties(self.pressure, self.temperature, self.specific_gravity)
        self.mu_gas = properties_gas.viscosity_gas()
        self.fvf_gas = properties_gas.factor_volumetric_gas()
        self.rho_gas = properties_gas.density_gas()
        self.z_gas = properties_gas.factor_compressibility_gas()
    
        properties_oil = OilProperties(self.pressure, self.temperature, self.specific_gravity, self.api, bubble_pressure=self.bubble_pressure)
        self.mu_oil = properties_oil.viscosity_oil()
        self.rho_oil = properties_oil.density_oil()
        self.sigma_oil = properties_oil.tension_oil()
        if self.sigma_oil < 1:
            self.sigma_oil = 1
        self.rs_oil = properties_oil.solution_oil()
        self.fvf_oil = properties_oil.factor_volumetric_oil()
        
        properties_water = WaterProperties(self.pressure, self.temperature, self.salinity)
        self.mu_water = properties_water.viscosity_water()
        self.rho_water = properties_water.density_water()
        self.sigma_water = properties_water.tension_water()
        if self.sigma_water < 1:
            self.sigma_water = 1
        self.rs_water = properties_water.solution_water()
        self.fvf_water = properties_water.factor_volumetric_water()
        
    def _properties_liquid_(self):
        self._properties_()
        self.rho_liq = self.rho_oil*(1/(1+self.wo_ratio)) + self.rho_water*(self.wo_ratio/(1+self.wo_ratio))
        self.mu_liq = self.mu_oil*(1/(1+self.wo_ratio)) + self.mu_water*(self.wo_ratio/(1+self.wo_ratio))
        self.sigma_liq = self.sigma_oil*(1/(1+self.wo_ratio)) + self.sigma_water*(self.wo_ratio/(1+self.wo_ratio))
    
    def _flow_(self):
        self._properties_()
        q_oil = (self.fvf_oil*self.qo_i)/15387
        q_water = (self.fvf_water*self.wo_ratio*self.qo_i)/15387
        q_liq = q_oil+q_water
        #if (self.go_ratio - self.rs_oil) < 0:
        #    q_gas = 0
        #else:
        #    q_gas = self.fvf_gas*(self.go_ratio-self.rs_oil-(self.rs_water*self.wo_ratio))*self.qo_i/86400
        q_gas = (self.fvf_gas*self.qo_i*self.go_ratio)/86400
        return [q_water, q_oil, q_gas, q_liq]
    
    def _velocities_(self):
        *_, qg, ql = self._flow_()
        v_sl = ql/self.area
        v_sg = qg/self.area
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
        self._properties_()
        self._properties_liquid_()
        Hl = self.holdup()
        return self.rho_liq*Hl+self.rho_gas*(1-Hl)
    
    def _number_reynolds_(self):
        *_, q_liq = self._flow_()
        Hl = self.holdup()
        M = self.sg_oil*350.52*(1/(1+self.wo_ratio)) + (self.rho_water/62.42)*350.52*(self.wo_ratio/(1+self.wo_ratio)) + self.specific_gravity*0.0764*self.gl_ratio
        NRe = 2.2e-2*((q_liq*15387*M)/((self.internal_diameter/12)*(self.mu_liq**Hl)*(self.mu_gas**(1-Hl))))
        if NRe < 4000:
            f = (-2*np.log10((1/3.7)*(self.rugosity/self.internal_diameter))+((6.81/NRe)**0.9))**(-2)
        else:
            f = 1/(-4*np.log10((self.rugosity/3.7065)-(5.0452/NRe*(np.log10((self.rugosity**(1.1098)/2.8257)+((7.149/NRe)**0.8991))))))**2
        return [NRe, f]
    
    def pressure_drop_friction(self):
        self._properties_()
        *_, q_liq = self._flow_()
        friction = self._number_reynolds_()[1]
        M = self.sg_oil*350.52*(1/(1+self.wo_ratio)) + (self.rho_water/62.42)*350.52*(self.wo_ratio/(1+self.wo_ratio)) + self.specific_gravity*0.0764*self.gl_ratio
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
            qoi.append(q_o*15387)
            qwi.append(q_w*15387)
            qgi.append(q_g*86400)
            qli.append(q_l*15387)
            
        return [np.array(qli), np.array(qgi), np.array(qoi), np.array(qwi), np.array(pwfi)]

    
if __name__ == "__main__":
    well = HagedornBrown(480, (100+460), bubble_pressure=1500)
    
    #print(well._sigma_tensions_())
    #print(well.holdup())
    #print(well._properties_oil_())
    #print(well._velocities_())
    #print(well._number_reynolds_())
    #print(well._rho_m_())
    #print(well.pressure_drop_gravity())
    #print(well.pressure_drop_friction())
    #print(well.bottom_hole_pressure())
    #print(well.outflow())
    #print(well.delta_t)
    
    #print(well.pressure_traverse_new())
    #print(well.outflow())
    #print(well._number_reynolds_())
    # qg = well.outflow()[1]
    # pwf = well.outflow()[4]
    # print("Qg", qg)
    # print("Pwf", pwf)
    
    #import matplotlib.pyplot as plt
    #
    #ql, qg, qo, qw, pw = well.outflow()
    #print(ql, qg, qo, qw, pw)
    #plt.plot(qo, pw)
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
    # plt.show()
    
    