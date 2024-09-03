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
                go_ratio: int|float=300, internal_diameter: int|float=2.5, well_depth: int|float=5000, 
                angle: int|float=90, qo_max: int|float=1000, amount: int=25):
        
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
        self.angle = angle
        self.qo_max = qo_max
        self.amount = amount
        
    def _properties_gas_(self):
        properties = GasProperties(self.pressure, self.temperature, self.specific_gravity)
        mu_gas = properties.viscosity_gas()
        fvf_gas = properties.factor_volumetric_gas()
        rho_gas = properties.density_gas()
        z_gas = properties.factor_compressibility_gas()
        return [mu_gas, rho_gas, fvf_gas, z_gas]
    
    def _properties_oil_(self):
        properties = OilProperties(self.pressure, self.temperature, self.specific_gravity, self.api, None, self.bubble_pressure)
        mu_oil = properties.viscosity_oil()
        rho_oil = properties.density_oil()
        sigma_oil = properties.tension_oil()
        if sigma_oil < 0:
            sigma_oil = np.array(1)
        rs_oil = properties.solution_oil()
        fvf_oil = properties.factor_volumetric_oil()
        return [mu_oil, rho_oil, sigma_oil, rs_oil, fvf_oil]
    
    def _properties_water_(self):
        properties = WaterProperties(self.pressure, self.temperature, self.salinity)
        mu_water = properties.viscosity_water()
        rho_water = properties.density_water()
        sigma_water = properties.tension_water()
        if sigma_water < 0:
            sigma_water = np.array(1)
        rs_water = properties.solution_water()
        fvf_water = properties.factor_volumetric_water()
        return [mu_water, rho_water, sigma_water, rs_water, fvf_water]
    
    def _properties_liquid_(self):
        mu_o, rho_o, sigma_o, rs_o, bo = self._properties_oil_()
        mu_w, rho_w, sigma_w, rs_w, bw = self._properties_water_()
        WOR = self.water_cut/(1-self.water_cut)
        rho_liq = rho_o*(1/(1+WOR)) + rho_w*(WOR/(1+WOR))
        mu_liq = mu_o*(1/(1+WOR)) + mu_w*(WOR/(1+WOR))
        sig_liq = sigma_o*(1/(1+WOR)) + sigma_w*(WOR/(1+WOR))
        return [rho_liq, mu_liq, sig_liq]
    
    def _flow_(self, qo=0.1):
        self.qo = qo
        *_, rs_o, bo = self._properties_oil_()
        *_, rs_w, bw = self._properties_water_()
        *_, bg, _ = self._properties_gas_()
        WOR = self.water_cut/(1-self.water_cut)
        #qo = np.linspace(1, self.qo_max, self.amount)
        q_oil = (bo*self.qo)/15387
        q_water = (bw*WOR*self.qo)/15387
        q_liq = q_oil+q_water
        if (self.go_ratio - rs_o) < 0:
            q_gas = 0
        else:
            q_gas = bg*(self.go_ratio-rs_o-(rs_w*WOR))*self.qo/86400
        
        return [q_water, q_oil, q_gas, q_liq]
    
    def _velocities_(self):
        *_, qg, ql = self._flow_(self.qo)
        area = np.pi/4*((self.internal_diameter/12)**2)
        v_sl = ql/area
        v_sg = qg/area
        v_m = v_sl + v_sg
        return [v_sl, v_sg, v_m]
    
    def _flow_regime_(self):
        rho_l, _, sig_l = self._properties_liquid_()
        vsl, _, vm = self._velocities_()
        Nfr = (vm**2)/((self.internal_diameter/12)*32.174)
        Nlv = 1.938*vsl*((rho_l/sig_l)**0.25)
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
    
    def holdup_liq(self):
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
            #if hold_liq_h > 1:
            #    hold_liq_h = 1
            holdup_liq = hold_liq_h*psi
            
            return holdup_liq
        #holdup_liquid = np.array([hold_l(nfr_val, nlv_val, lambl_val, regime_val, l2_val, l3_val) for nfr_val, nlv_val, lambl_val, regime_val, l2_val, l3_val in zip(nfr, nlv, lamb_liq, regime, l2, l3)])
        holdup_liquid = hold_l(nfr, nlv, lamb_liq, regime, l2, l3)
        holdup_gas = 1 - holdup_liquid
        
        return [holdup_liquid, holdup_gas]        
        
    def _properties_mixture_(self):
        *_, lmbd_l, lmbd_g = self._flow_regime_()
        h_l, h_g = self.holdup_liq()
        rho_l, mu_l, _ = self._properties_liquid_()
        mu_g, rho_g, *_ = self._properties_gas_()
        rho_m = rho_l*lmbd_l + rho_g*lmbd_g
        mu_m = mu_l*lmbd_l + mu_g*lmbd_g
        rho_mis = rho_l*h_l + rho_g*h_g
        return [rho_m, mu_m, rho_mis]
    
    def _number_reynolds_(self, rugosity=0.0001):
        self.rugosity = rugosity
        *_, v_m = self._velocities_()
        rho_m, mu_m, _ = self._properties_mixture_()
        lmbd_l = self._flow_regime_()[-2]
        h_l = self.holdup_liq()[0]
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
        
    def _delta_depth_(self):
        return np.linspace(0, self.well_depth, self.amount)
    
    def _delta_temp_(self, bh_temperature: int|float=600):
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
            #ql, qg, qo, qw, pwf
        return [np.array(qli), np.array(qgi), np.array(qoi), np.array(qwi), np.array(pwf_list)]
        
            
if __name__ == "__main__":
    
    well = BeggsBrill(450, (100+460), bubble_pressure=1500)
    #well._flow_(500)
    #print(well._delta_temp_())
    #print(well._delta_depth_())
    #print(well._properties_oil_())
    #print(well._flow_())
    #print(well._velocities_())
    #print(well._flow_regime_())
    #print(well._regime_())
    #print(well.holdup_liq())
    
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