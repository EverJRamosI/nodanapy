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
    
        properties_oil = OilProperties(self.pressure, self.temperature, self.specific_gravity, self.api, None, self.bubble_pressure)
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
        self.sig_liq = self.sigma_oil*(1/(1+self.wo_ratio)) + self.sigma_water*(self.wo_ratio/(1+self.wo_ratio))
    
    def _flow_(self):
        self._properties_()
        q_oil = (self.fvf_oil*self.qo_i)/15387
        q_water = (self.fvf_water*self.wo_ratio*self.qo_i)/15387
        q_liq = q_oil+q_water
        #if (self.go_ratio - rs_o) < 0:
        #    q_gas = 0
        #else:
        #    q_gas = bg*(self.go_ratio-rs_o-(rs_w*WOR))*self.qo/86400
        q_gas = (self.fvf_gas*self.qo_i*self.go_ratio)/86400
        return [q_water, q_oil, q_gas, q_liq]
    
    def _velocities_(self):
        *_, qg, ql = self._flow_()
        v_sl = ql/self.area
        v_sg = qg/self.area
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
        self._properties_()
        self._properties_liquid_()
        *_, lmbd_l, lmbd_g = self._flow_regime_()
        h_l = self.holdup()
        rho_m = self.rho_liq*lmbd_l + self.rho_gas*lmbd_g
        mu_m = self.mu_liq*lmbd_l + self.mu_gas*lmbd_g
        rho_mis = self.rho_liq*h_l + self.rho_gas*(1-h_l)
        return [rho_m, mu_m, rho_mis]
    
    def _number_reynolds_(self):
        *_, v_m = self._velocities_()
        rho_m, mu_m, _ = self._properties_mixture_()
        lmbd_l = self._flow_regime_()[-2]
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
        angle_value = (self.angle*np.pi)/180
        rho_mis = self._properties_mixture_()[-1]
        delta_pressure = (rho_mis*np.sin(angle_value))/144
        return delta_pressure
    
    def pressure_drop_friction(self):
        rho_mis = self._properties_mixture_()[-1]
        v_m = self._velocities_()[2]
        ft = self._number_reynolds_()[1]
        delta_pressure = 2*ft*rho_mis*(v_m**2) / 32.17 / (self.internal_diameter/12) / 144
        return delta_pressure
    
    def kinetic_factor(self):
        v_sl, v_sg, v_m = self._velocities_()
        rho_mis = self._properties_mixture_()[-1]
        Ek = v_m*v_sg*rho_mis / 32.17 / self.pressure / 144
        return Ek
    
    def pressure_drop_total(self):
        dp_total = (self.pressure_drop_potential() + self.pressure_drop_friction()) / (1 - self.kinetic_factor())
        return dp_total
    
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
    
    well = BeggsBrill(450, (100+460), bubble_pressure=1500, water_cut=0.5)
    #well._flow_(500)
    #print(well._delta_temp_())
    #print(well._delta_depth_())
    #print(well._properties_oil_())
    #print(well._flow_())
    #print(well._velocities_())
    #print(well._flow_regime_())
    #print(well._regime_())
    #print(well.holdup())
    
    #print(well._delta_flow_())
    
    #print(well._number_reynolds_())
    #print(well.pressure_drop_friction())
    #print(well.pressure_drop_total())
    
    #print(well.pressure_traverse())
    
    #dp = well._pressure_traverse_iter_()[0]
    #dh = well._pressure_traverse_iter_()[3]
    #print(dp, dh)
    #q = well._delta_flow_()
    #p = well.bottom_hole_pressure()
    #print(q, p)
    #print(well.outflow())
    
    import matplotlib.pyplot as plt
    #
    ql, qg, qo, qw, pw = well.outflow()
    print(ql, qg, qo, qw, pw)
    plt.plot(qo, pw)
    plt.show()
    
    #h = well.delta_depth
    #p, dp, hl = well.pressure_traverse()
    #print(p, dp, hl, h)
    #fig, ax = plt.subplots(1, 3)
    #ax[0].invert_yaxis()
    #ax[1].invert_yaxis()
    #ax[2].invert_yaxis()
    #ax[0].plot(p, h)
    #ax[1].plot(dp, h)
    #ax[2].plot(hl, h)
    #plt.show()
    