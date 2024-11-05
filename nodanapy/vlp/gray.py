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
                go_ratio: int|float=5000, internal_diameter: int|float=2.5, rugosity: float=0.0001, well_depth: int|float=5000, 
                temperature_node: int|float=600, ql_i: int|float=0.001, ql_n: int|float=1000, amount: int=25):
        
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
        self.ql_i = ql_i
        self.ql_n = ql_n
        self.amount = amount
        
        self.wo_ratio = self.water_cut/(1-self.water_cut)
        self.sg_oil = 141.5/(131.5+self.api)
        self.gl_ratio = (self.go_ratio)*(1-self.water_cut)
        self.area = np.pi/4*((self.internal_diameter/12)**2)
                
        self.delta_ql = np.linspace(self.ql_i, self.ql_n, self.amount)
        self.delta_depth = np.linspace(0, self.well_depth, self.amount)
        self.delta_t = self._delta_temp_()
        
        self._prop_gas = GasProperties(self.pressure, self.temperature, self.specific_gravity)
        self._prop_oil = OilProperties(self.pressure, self.temperature, self.specific_gravity, self.api, bubble_pressure=self.bubble_pressure)
        self._prop_water = WaterProperties(self.pressure, self.temperature, self.salinity)
        self._q_water, self._q_oil, self._q_gas, self._q_liq = self._flow_()
        self._v_sg, self._v_sl, self._v_m = self._velocities_()
        self._f_oil, self._f_water = self._fractions_liquid_()
        self._lambda_liq, self._lambda_gas = self._no_slip_holdup_()
        self._rho_liq, self._rho_ns, self._sigma_liq, self._mu_liq, self._mu_ns = self._properties_liquid_()
        self._holdup = self.holdup()
        self._rho_m = self._rho_m_()
        self._NRe, self._f = self._number_reynolds_()
    
    
    def _delta_temp_(self):
        gradient = np.abs((self.temperature-self.temperature_node))/self.well_depth
        return self.temperature + gradient*self.delta_depth
        
    def _flow_(self):
        q_liq = self.ql_i
        q_oil = (1-self.water_cut)*self.ql_i
        q_gas = self.ql_i*self.gl_ratio
        q_water = self.water_cut*self.ql_i
        return [q_water, q_oil, q_gas, q_liq]

    def _velocities_(self):
        v_sl = (5.615*self._q_liq/(86400*self.area))*(self._prop_oil.factor_volumetric_oil()*(1/(1+self.wo_ratio)) + self._prop_water.factor_volumetric_water()*(self.wo_ratio/(1+self.wo_ratio)))
        v_sg = (self._q_gas/(86400*self.area))*self._prop_gas.factor_volumetric_gas()
        v_m = v_sg + v_sl
        return [v_sg, v_sl, v_m]
    
    def _fractions_liquid_(self):
        f_oil = self._q_oil/self._q_liq
        f_water = 1-f_oil
        return [f_oil, f_water]        
    
    def _no_slip_holdup_(self):
        lambda_liq = self._v_sl/self._v_m
        lambda_gas = 1-lambda_liq
        return [lambda_liq, lambda_gas]
    
    def _properties_liquid_(self):
        rho_liq = self._prop_oil.density_oil()*self._f_oil+self._prop_water.density_water()*self._f_water
        rho_ns = rho_liq*self._lambda_liq+self._prop_gas.density_gas()*self._lambda_gas
        
        sigma_oil = self._prop_oil.tension_oil()
        if sigma_oil < 1:
            sigma_oil = 1
        sigma_water = self._prop_water.tension_water()
        if sigma_water < 1:
            sigma_water = 1
        sigma_liq = (self._f_oil*sigma_oil+0.617*self._f_water*sigma_water)/(self._f_oil+0.617*self._f_water)

        mu_liq = self._prop_water.viscosity_water()*self._f_water + self._prop_oil.viscosity_oil()*self._f_oil
        mu_ns = self._prop_gas.viscosity_gas()*self._lambda_gas+mu_liq*self._lambda_liq
        return [rho_liq, rho_ns, sigma_liq, mu_liq, mu_ns]
    
    def holdup(self):
        N1 = 453.592*(((self._rho_ns**2)*(self._v_m**4))/(32.17*self._sigma_liq*(self._rho_liq-self._prop_gas.density_gas())))
        N2 = 453.592*((32.17*(self._rho_liq-self._prop_gas.density_gas())*((self.internal_diameter/12)**2))/(self._sigma_liq))
        R_v = self._v_sl/self._v_sg
        N3 = 0.0814*(1-0.0554*np.log(1+((730*R_v)/(R_v+1))))
        G = -2.314*((N1*(1+(205/N2)))**N3)
        holdup_liq = 1-((1-np.exp(G))/(R_v+1))
        return holdup_liq
    
    def _rho_m_(self):
        return self._holdup*self._rho_liq+self._prop_gas.density_gas()*(1-self._holdup)
    
    def _number_reynolds_(self):
        NRe = (self._rho_ns*self._v_m*(self.internal_diameter/12))/(0.000672*self._mu_ns)
        R_v = self._v_sl/self._v_sg
        rugosity_o = (28.5*self._sigma_liq)/(453.592*self._rho_ns*((self._v_m)**2))
        
        if R_v < 0.007:
            rugosity = self.rugosity + R_v*((rugosity_o - self.rugosity)/0.007)
        else:
            rugosity = rugosity_o
        
        if NRe < 4000:
            f = (-2*np.log10((1/3.7)*(rugosity/self.internal_diameter))+((6.81/NRe)**0.9))**(-2)
        else:
            f = 1/(-4*np.log10((rugosity/3.7065)-(5.0452/NRe*(np.log10((rugosity**(1.1098)/2.8257)+((7.149/NRe)**0.8991))))))**2
        
        if f>1:
            if NRe < 4000:
                f = (-2*np.log10((1/3.7)*(self.rugosity/self.internal_diameter))+((6.81/NRe)**0.9))**(-2)
            else:
                f = 1/(-4*np.log10((self.rugosity/3.7065)-(5.0452/NRe*(np.log10((self.rugosity**(1.1098)/2.8257)+((7.149/NRe)**0.8991))))))**2
        
        return [NRe, f]
    
    def pressure_traverse(self):
        p = [self.pressure]
        vsl = [self._v_sl]
        vsg = [self._v_sg]
        vm = [self._v_m]
        hl = [self._holdup]
        dpf_i = (self._rho_ns*self._f*(self._v_m**2))/(2*32.17*(self.internal_diameter/12))
        dpt_i = (dpf_i + self._rho_m)/144
        dp = [dpt_i]
        
        for dh, ti, i in zip(self.delta_depth, self.delta_t, range(self.amount)):
            
            if i == 0:
                continue
            
            pi = p[i-1]+dp[i-1]*(dh-self.delta_depth[i-1])
            self._prop_gas = GasProperties(pi, ti, self.specific_gravity)
            self._prop_oil = OilProperties(pi, ti, self.specific_gravity, self.api, bubble_pressure=self.bubble_pressure)
            self._prop_water = WaterProperties(pi, ti, self.salinity)
            self._v_sg, self._v_sl, self._v_m = self._velocities_()
            self._f_oil, self._f_water = self._fractions_liquid_()
            self._lambda_liq, self._lambda_gas = self._no_slip_holdup_()
            self._rho_liq, self._rho_ns, self._sigma_liq, self._mu_liq, self._mu_ns = self._properties_liquid_()
            self._holdup = self.holdup()
            self._rho_m = self._rho_m_()
            self._NRe, self._f = self._number_reynolds_()
            
            dpf = self._rho_ns*((self._f*(self._v_m**2))/(2*32.17*(self.internal_diameter/12)))
            dph = self._rho_ns*(((self._v_m**2)/(2*32.17))/(dh-self.delta_depth[i-1]))
            dpt = (self._rho_m + dpf + dph)/144
            pj = p[i-1]+dpt*(dh-self.delta_depth[i-1])
            
            vsl.append(self._v_sl)
            vsg.append(self._v_sg)
            vm.append(self._v_m)
            hl.append(self._holdup)
            
            dp.append(dpt)
            p.append(pj)
            
        return [np.array(vsl), np.array(vsg), np.array(vm), np.array(hl), np.array(dp), np.array(p)]
    
    def outflow(self):
        qoi = []
        qwi = []
        qgi = []
        qli = []
        pwfi = []
        
        for flow_value in self.delta_ql:
            self.ql_i = flow_value
            self._q_water, self._q_oil, self._q_gas, self._q_liq = self._flow_()
            *_, pwf = self.pressure_traverse()
            pwfi.append(pwf[-1])
            qoi.append(self._q_oil)
            qwi.append(self._q_water)
            qgi.append(self._q_gas/1000)
            qli.append(self._q_liq)
            
        return [np.array(qli), np.array(qgi), np.array(qoi), np.array(qwi), np.array(pwfi)]

if __name__ == "__main__":
    # import time
    # time_start = time.time()
    well1 = Gray(149.7, (84+460), temperature_node=(140+460), go_ratio=5000, internal_diameter=2.5, water_cut=0.85, ql_n=1000) #ql_i=0.001, ql_n=5.06801)# 0.673, 53.7, 149.7, 8415, 0.88, 48.5981, 2.441, 5098, 7800, 5)
    # well2 = Gray(149.7, (84+460), temperature_node=(140+460), go_ratio=5000, internal_diameter=0.5, water_cut=0.85, ql_n=1000) #ql_i=0.001, ql_n=5.06801)# 0.673, 53.7, 149.7, 8415, 0.88, 48.5981, 2.441, 5098, 7800, 5)
    # well3 = Gray(149.7, (84+460), temperature_node=(140+460), go_ratio=5000, internal_diameter=0.75, water_cut=0.85, ql_n=1000) #ql_i=3.28075, ql_n=26.239

    # import matplotlib.pyplot as plt

    # h = well2.delta_depth
    # vl, vg, vm, hl, dp, p = well2.pressure_traverse()
    # print('vl', vl, 'vg', vg, 'vm', vm, 'hl', hl, 'dp', dp, 'P', p)
    # fig, ax = plt.subplots(2, 3)
    
    # ax[0, 0].invert_yaxis()
    # ax[0, 0].plot(dp, h)
    
    # ax[0, 1].invert_yaxis()
    # ax[0, 1].plot(p, h)
    
    # ax[0, 2].invert_yaxis()
    # ax[0, 2].plot(hl, h)
    
    # ax[1, 0].invert_yaxis()
    # ax[1, 0].plot(vl, h)
    
    # ax[1, 1].invert_yaxis()
    # ax[1, 1].plot(vg, h)
    
    # ax[1, 2].invert_yaxis()
    # ax[1, 2].plot(vm, h)
    # plt.show()
    # ql1, qg1, qo1, qw1, pw1 = well1.outflow()
    # ql2, qg2, qo2, qw2, pw2 = well2.outflow()
    # ql3, qg3, qo3, qw3, pw3 = well3.outflow()
    # print('Well1')
    # print(ql1, pw1, qg1)
    # print('Well2')
    # print(ql2, pw2, qg2)
    # print('Well3')
    # print(ql3, pw3, qg3)
    # plt.plot(ql1, pw1)
    # plt.plot(ql2, pw2)
    # plt.plot(ql3, pw3)
    # time_end = time.time()
    # print('time', time_end-time_start)
    # plt.show()