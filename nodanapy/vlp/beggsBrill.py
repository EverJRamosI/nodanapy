import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties
from _properties.oilProperties import OilProperties
from _properties.waterProperties import WaterProperties

class BeggsBrill:
    def __init__(self, pressure: int|float, temperature: int|float, specific_gravity: float=0.65, 
                api: int|float=40, bubble_pressure: int|float=0, salinity: int|float=1000, water_cut: float=0.0, 
                go_ratio: int|float=300, internal_diameter: int|float=2.5, rugosity: float=0.0001, well_depth: int|float=5000, 
                temperature_node: int|float=600, angle: int|float=90, qo_i: int|float=0.01, qo_n: int|float=1000, amount: int=25):
        
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
        self.angle = angle
        self.qo_i = qo_i
        self.qo_n = qo_n
        self.amount = amount
        
        self.wo_ratio = self.water_cut/(1-self.water_cut)
        self.sg_oil = 141.5/(131.5+self.api)
        self.gl_ratio = (self.go_ratio)/(self.wo_ratio+1)
        self.area = np.pi/4*((self.internal_diameter/12)**2)
        self.angle_value = (self.angle*np.pi)/180
        
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
        self.sig_liq = sigma_oil*(1/(1+self.wo_ratio)) + sigma_water*(self.wo_ratio/(1+self.wo_ratio))
    
    def _flow_(self):
        #q_oil = (self._prop_oil.factor_volumetric_oil()*self.qo_i)/15387
        q_oil = self.qo_i
        #q_water = (self._prop_water.factor_volumetric_water()*self.wo_ratio*self.qo_i)/15387
        q_water = self.wo_ratio*self.qo_i
        q_liq = q_oil+q_water
        #if (self.go_ratio - rs_o) < 0:
        #    q_gas = 0
        #else:
        #    q_gas = bg*(self.go_ratio-rs_o-(rs_w*WOR))*self.qo/86400
        #q_gas = (self._prop_gas.factor_volumetric_gas()*self.qo_i*self.go_ratio)/15387
        q_gas = self.qo_i*self.go_ratio
        return [q_water, q_oil, q_gas, q_liq]
    
    def _velocities_(self):
        qw, qo, qg, ql = self._flow_()
        #v_sl = ql/self.area
        v_sl = (((self._prop_oil.factor_volumetric_oil()*qo)/15387) + ((self._prop_water.factor_volumetric_water()*qw)/15387))/self.area
        #v_sg = qg/self.area
        v_sg = ((self._prop_gas.factor_volumetric_gas()*qg)/15387)/self.area
        v_m = v_sl + v_sg
        return [v_sl, v_sg, v_m]
    
    def _flow_regime_(self):
        self._properties_liquid_()
        vsl, _, vm = self._velocities_()
        Nfr = (vm**2)/((self.internal_diameter/12)*32.174)
        Nlv = 1.938*vsl*((self.rho_liq/self.sig_liq)**0.25)
        lambda_liq = vsl / vm
        lambda_gas = 1-lambda_liq
        L1 = 316*(lambda_liq**0.302)
        L2 = 9.252e-4*(lambda_liq**(-2.4684))
        L3 = 0.10*(lambda_liq**(-1.4516))
        L4 = 0.5*(lambda_liq**(-6.738))
        return [Nfr, Nlv, L1, L2, L3, L4, lambda_liq, lambda_gas]
    
    def _regime_(self):
        nfr, _, l1, l2, l3, l4, lambl, _ = self._flow_regime_()
        
        def reg(nfr_v, lambl_v, l1_v, l2_v, l3_v, l4_v):
            if ((lambl_v<0.01 and nfr_v<1) or (lambl_v>=0.01 and nfr_v<l2_v)):
                return 'segregated flow'
            elif (lambl_v>=0.01 and l2_v<nfr_v<=l3_v):
                return 'transition flow'
            elif ((0.01<=lambl_v<0.4 and l3_v<nfr_v<=l1_v) or (lambl_v>=0.4 and l3_v<nfr_v<=l4_v)):
                return 'intermittent flow'
            elif ((lambl_v<0.4 and nfr_v>=l1_v) or (lambl_v>=0.4 and nfr_v>l4_v)):
                return 'distributed flow'
        #regime = [reg(nfr_val, lamb_val, l1_val, l2_val, l3_val, l4_val) for nfr_val, lamb_val, l1_val, l2_val, l3_val, l4_val in zip(nfr, lambl, l1, l2, l3, l4)]
        regime = reg(nfr, lambl, l1, l2, l3, l4)
        return regime
    
    def holdup(self):
        regime = self._regime_()
        nfr, nlv, _, l2, l3, _, lamb_liq, _ = self._flow_regime_()
        angle_value = (self.angle*np.pi)/180
        
        def hold_l(nfr_v, nlv_v, lambl_v, regime_v, l2_v, l3_v):
            if regime_v == 'segregated flow':
                a, b, c = 0.98, 0.4846, 0.0868
                if angle_value >= 0:
                    d, e, f, g = 0.011, -3.768, 3.539, -1.614
                else:
                    d, e, f, g = 4.7, -0.3692, 0.1244, -0.5056
            elif regime_v == 'transition flow':
                h0 = (l3_v - nfr_v)/(l3_v - l2_v)
                hold_segre = hold_l(nfr_v, nlv_v, lambl_v, 'segregated flow', l2_v, l3_v)
                hold_inter = hold_l(nfr_v, nlv_v, lambl_v, 'intermittent flow', l2_v, l3_v)
                return h0*hold_segre + (1 - h0)*hold_inter                
            elif regime_v == 'intermittent flow':
                a, b, c = 0.845, 0.5351, 0.0173
                if angle_value >= 0:
                    d, e, f, g = 0.011, -3.768, 3.539, -1.614
                else:
                    d, e, f, g = 4.7, -0.3692, 0.1244, -0.5056
            elif regime_v == 'distributed flow':
                a, b, c = 1.065, 0.5824, 0.0609
                if angle_value >= 0:
                    d, e, f, g = 1, 0, 0, 0
                else:
                    d, e, f, g = 4.7, -0.3692, 0.1244, -0.5056
                    
            cof_incl = (1-lambl_v)*np.log(d*(lambl_v**e)*(nlv_v**f)*(nfr_v**g))
            
            if cof_incl < 0:
                cof_incl = 0
                
            psi = 1 + cof_incl*(np.sin(1.8*angle_value) - 0.333*(np.sin(1.8*angle_value)**3))
            
            hold_liq_h = (a*(lambl_v**b))/(nfr_v**c)
                
            if hold_liq_h < lambl_v:
                hold_liq_h = lambl_v
            #if hold_liq_h > 1.0:
            #    hold_liq_h = 1.0
            holdup_liq = hold_liq_h*psi
            
            if holdup_liq > 1.0:
                holdup_liq = 1.0
            
            return holdup_liq
        #holdup_liquid = np.array([hold_l(nfr_val, nlv_val, lambl_val, regime_val, l2_val, l3_val) for nfr_val, nlv_val, lambl_val, regime_val, l2_val, l3_val in zip(nfr, nlv, lamb_liq, regime, l2, l3)])
        holdup_liquid = hold_l(nfr, nlv, lamb_liq, regime, l2, l3)
        return holdup_liquid       
    
    def _properties_mixture_(self):
        self._properties_liquid_()
        *_, lmbd_l, lmbd_g = self._flow_regime_()
        h_l = self.holdup()
        rho_m = self.rho_liq*lmbd_l + self._prop_gas.density_gas()*lmbd_g
        mu_m = self.mu_liq*lmbd_l + self._prop_gas.viscosity_gas()*lmbd_g
        rho_mis = self.rho_liq*h_l + self._prop_gas.density_gas()*(1-h_l)
        return [rho_m, mu_m, rho_mis]
    
    def _number_reynolds_(self):
        *_, v_m = self._velocities_()
        rho_m, mu_m, _ = self._properties_mixture_()
        *_, lmbd_l, _ = self._flow_regime_()
        h_l = self.holdup()
        NRe = (1488*rho_m*v_m*(self.internal_diameter/12))/(mu_m)
        fn = (-2*np.log10((1/3.7)*(self.rugosity/self.internal_diameter))+((6.81/NRe)**0.9))**(-2)
        y = lmbd_l/(h_l**2)
        
        def factor_t(y_v, fn_v):
            if (y_v>1 and y_v<1.2):
                s = np.log(2.2*y_v - 1.2)
            else:
                s = np.log(y_v)/((-0.0523) + (3.182*np.log(y_v)) - (0.8725*((np.log(y_v))**2)) + (0.01853*((np.log(y_v))**4)))

            return fn_v*np.exp(s)
        #ft = np.array([factor_t(y_val, fn_val) for y_val, fn_val in zip(y, fn)])
        ft = factor_t(y, fn)
        return [NRe, ft]
    
    def pressure_drop_potential(self):
        rho_mis = self._properties_mixture_()[-1]
        return (rho_mis*np.sin(self.angle_value))/144
    
    def pressure_drop_friction(self):
        rho_mis = self._properties_mixture_()[-1]
        v_m = self._velocities_()[2]
        ft = self._number_reynolds_()[1]
        return 2*ft*rho_mis*(v_m**2) / 32.17 / (self.internal_diameter/12) / 144
    
    def kinetic_factor(self):
        v_sl, v_sg, v_m = self._velocities_()
        rho_mis = self._properties_mixture_()[-1]
        return v_m*v_sg*rho_mis / 32.17 / self.pressure / 144
    
    def pressure_drop_total(self):
        return (self.pressure_drop_potential() + self.pressure_drop_friction()) / (1 - self.kinetic_factor())
    
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
    #import time
    #time_start = time.time()
    well = BeggsBrill(450, (100+460), bubble_pressure=1500)
    
    #print(well.pressure_traverse())
    
    #print(well.outflow())
    import matplotlib.pyplot as plt
    
    # ql, qg, qo, qw, pw = well.outflow()
    # print(ql, qg, qo, qw, pw)
    # plt.plot(qo, pw)
    # plt.show()
    
    h = well.delta_depth
    p, dp, hl = well.pressure_traverse()
    print(p, dp, hl, h)
    # fig, ax = plt.subplots(1, 3)
    # ax[0].invert_yaxis()
    # ax[1].invert_yaxis()
    # ax[2].invert_yaxis()
    # ax[0].plot(p, h)
    # ax[1].plot(dp, h)
    # ax[2].plot(hl, h)
    
    # time_end = time.time()
    # print('time', time_end-time_start)
    # plt.show()
    