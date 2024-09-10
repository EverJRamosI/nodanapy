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
                go_ratio: int|float=300, internal_diameter: int|float=2.5, well_depth: int|float=5000, 
                qo_max: int|float=1000, amount: int=25):
        
        self.pressure = pressure
        self.temperature = temperature
        self.specific_gravity = specific_gravity
        self.api = api
        self.bubble_pressure = bubble_pressure
        self.salinity = salinity
        self.water_cut = water_cut
        self.go_ratio = go_ratio
        self.internal_diameter = internal_diameter
        self.well_depth = well_depth
        self.qo_max = qo_max
        self.amount = amount
        
        self.WOR = self.water_cut/(1-self.water_cut)
        self.sg_oil = 141.5/(131.5+self.api)
        self.GLR = (self.go_ratio)/(self.WOR+1)
        
    def _properties_gas_(self):
        properties = GasProperties(self.pressure, self.temperature, self.specific_gravity)
        mu_gas = properties.viscosity_gas()
        fvf_gas = properties.factor_volumetric_gas()
        rho_gas = properties.density_gas()
        z_gas = properties.factor_compressibility_gas()
        return [mu_gas, rho_gas, fvf_gas, z_gas]
    
    def _properties_oil_(self):
        properties = OilProperties(self.pressure, self.temperature, self.specific_gravity, self.api, self.bubble_pressure)
        mu_oil = properties.viscosity_oil()
        rho_oil = properties.density_oil()
        sigma_oil = properties.tension_oil()
        if sigma_oil < 0:
            sigma_oil = 1
        rs_oil = properties.solution_oil()
        fvf_oil = properties.factor_volumetric_oil()
        return [mu_oil, rho_oil, sigma_oil, rs_oil, fvf_oil]
    
    def _properties_water_(self):
        properties = WaterProperties(self.pressure, self.temperature, self.salinity)
        mu_water = properties.viscosity_water()
        rho_water = properties.density_water()
        sigma_water = properties.tension_water()
        if sigma_water < 0:
            sigma_water = 1
        rs_water = properties.solution_water()
        fvf_water = properties.factor_volumetric_water()
        return [mu_water, rho_water, sigma_water, rs_water, fvf_water]
    
    def _properties_liquid_(self):
        mu_o, rho_o, sigma_o, *_ = self._properties_oil_()
        mu_w, rho_w, sigma_w, *_ = self._properties_water_()
        rho_liq = rho_o*(1/(1+self.WOR)) + rho_w*(self.WOR/(1+self.WOR))
        mu_liq = mu_o*(1/(1+self.WOR)) + mu_w*(self.WOR/(1+self.WOR))
        sig_liq = sigma_o*(1/(1+self.WOR)) + sigma_w*(self.WOR/(1+self.WOR))
        return [rho_liq, mu_liq, sig_liq]
    
    def _flow_(self, qo=0.1):
        self.qo = qo
        *_, rs_o, bo = self._properties_oil_()
        *_, rs_w, bw = self._properties_water_()
        *_, bg, _ = self._properties_gas_()
        q_oil = (bo*self.qo)/15387
        q_water = (bw*self.WOR*self.qo)/15387
        q_liq = q_oil+q_water
        if (self.go_ratio - rs_o) < 0:
            q_gas = 0
        else:
            q_gas = bg*(self.go_ratio-rs_o-(rs_w*self.WOR))*self.qo/86400
        
        return [q_water, q_oil, q_gas, q_liq]
    
    def _velocities_(self):
        *_, qg, ql = self._flow_(self.qo)
        area = np.pi/4*((self.internal_diameter/12)**2)
        v_sl = ql/area
        v_sg = qg/area
        v_m = v_sl + v_sg
        return [v_sl, v_sg, v_m]
    
    def _holdup_(self):
        rho_l, mu_l, sig_l = self._properties_liquid_()
        vsl, vsg, vm = self._velocities_()
        
        A = 1.071 - ((0.2218*((vm)**2))/(self.internal_diameter/12))
        
        if A < 0.13:
            A = 0.13
        
        B = vsg/vm
        
        if B > A:
            NLV = 1.938*vsl*((rho_l/sig_l)**(1/4))
            NGV = 1.938*vsg*((rho_l/sig_l)**(1/4))
            ND = 120.872*(self.internal_diameter/12)*(np.sqrt(rho_l/sig_l))
            NL = 0.15726*mu_l*((1/(rho_l*(sig_l**3)))**(1/4))
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

        return holdup_liq
    
    def _rho_m_(self):
        rho_l, *_ = self._properties_liquid_()
        _, rho_g, *_ = self._properties_gas_()
        Hl = self._holdup_()
        return rho_l*Hl+rho_g*(1-Hl)
    
    def _delta_depth_(self):
        return np.linspace(0, self.well_depth, self.amount)
                    
    def pressure_drop_gravity(self):
        dH = np.diff(self._delta_depth_())
        vm = self._velocities_()[2]
        delta_pressure = self._rho_m_()*(((vm**2)/(2*32.17))/dH[0])
        return delta_pressure
    
    def _number_reynolds_(self, rugosity=0.0001):
        self.rugosity = rugosity
        *_, q_liq = self._flow_(self.qo)
        rho_w = self._properties_water_()[1]
        mu_l = self._properties_liquid_()[1]
        mu_g = self._properties_gas_()[0]
        Hl = self._holdup_()
        M = self.sg_oil*350.52*(1/(1+self.WOR)) + (rho_w/62.42)*350.52*(self.WOR/(1+self.WOR)) + self.specific_gravity*0.0764*self.GLR
        NRe = 2.2e-2*((q_liq*M)/((self.internal_diameter/12)*(mu_l**Hl)*(mu_g**(1-Hl))))
        f = (-2*np.log10((1/3.7)*(self.rugosity/self.internal_diameter))+((6.81/NRe)**0.9))**(-2)
        
        return [NRe, f]
    
    def pressure_drop_friction(self):
        *_, q_liq = self._flow_(self.qo)
        friction = self._number_reynolds_()[1]
        rho_w = self._properties_water_()[1]
        M = self.sg_oil*350.52*(1/(1+self.WOR)) + (rho_w/62.42)*350.52*(self.WOR/(1+self.WOR)) + self.specific_gravity*0.0764*self.GLR
        delta_pressure = (friction*(M**2)*((q_liq*15387)**2))/(7.413e10*((self.internal_diameter/12)**5)*self._rho_m_())
        return delta_pressure
    
    def pressure_drop_total(self):
        return (self.pressure_drop_friction() + self._rho_m_())/144    
    
    def _delta_temp_(self, bh_temperature=600):
        self.bh_temperature = bh_temperature
        gradient = np.abs((self.temperature-self.bh_temperature))/self.well_depth
        return self.temperature + gradient*self._delta_depth_()
    
    def _delta_flow_(self, qo=0.1):
        #self.qo = qo
        q_oil = np.linspace(qo, self.qo_max, self.amount)
        return q_oil
    
    def pressure_traverse(self):
        pn = self.pressure
        dT = self._delta_temp_()
        dH = self._delta_depth_()
        
        p = [pn]
        dpt = [self.pressure_drop_total()]

        dz_array = np.diff(dH)

        for i, dz in enumerate(dz_array, 1):
            self.temperature = dT[i]

            pi = dpt[i-1] * dz + p[i-1]
            self.pressure = pi
            dP_n = self.pressure_drop_total()
            
            p.append(pi)
            dpt.append(dP_n)
            
        self.temperature = dT[0]
        self.pressure = pn
        
        return [np.array(p), np.array(dpt), dT, dH]
    
    def outflow(self):
        qon = self._delta_flow_()    
        qoi = []
        qwi = []
        qgi = []
        qli = []
        pwf_list = []
        
        for flow_value in qon:
            q_w, q_o, q_g, q_l = self._flow_(flow_value)
            pwf, *_ = self.pressure_traverse()
            pwf_list.append(pwf[-1])
            qoi.append(q_o*15387)
            qwi.append(q_w*15387)
            qgi.append(q_g*86400)
            qli.append(q_l*15387)
            
        return [np.array(qli), np.array(qgi), np.array(qoi), np.array(qwi), np.array(pwf_list)]

    
if __name__ == "__main__":
    well = HagedornBrown(480, (100+460), bubble_pressure=1500,)
    
    #print(well._sigma_tensions_())
    #print(well._holdup_())
    #print(well._properties_oil_())
    #print(well._velocities_())
    #print(well._number_reynolds_())
    #print(well._rho_m_())
    #print(well.pressure_drop_gravity())
    #print(well.pressure_drop_friction())
    #print(well.bottom_hole_pressure())
    #print(well.outflow())
    
    #print(well.pressure_traverse())
    
    # qg = well.outflow()[1]
    # pwf = well.outflow()[4]
    # print("Qg", qg)
    # print("Pwf", pwf)
    
    import matplotlib.pyplot as plt
    #
    ql, qg, qo, qw, pw = well.outflow()
    print(ql, qg, qo, qw, pw)
    plt.plot(qo, pw)
    plt.show()
    
    
    # plt.plot(qg, pwf)
    # plt.show()