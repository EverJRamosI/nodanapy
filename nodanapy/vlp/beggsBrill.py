import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties
from _properties.oilProperties import OilProperties
from _properties.waterProperties import WaterProperties

class BeggsBrill:
    """
    ### Summary:
    This class is to determine VLP with the Beggs and Brill method.
    
    ### Methods:
    - __delta_temp_
    - __flow_
    - __properties_liquid_
    - __total_mass_
    - __velocities_
    - __flow_regime_
    - __regime_
    - holdup: This method determines the holdup
    - __properties_mixture_
    - __number_reynolds_
    - pressure_traverse: This method determines the flowing pressure
    - outflow: This method is to calculate the VLP.
    """  
    def __init__(self, pressure: int|float, temperature: int|float, specific_gravity: float=0.65, 
                api: int|float=40, bubble_pressure: int|float=0, salinity: int|float=1000, water_cut: float=0.0, 
                go_ratio: int|float=300, internal_diameter: int|float=2.5, rugosity: float=0.0001, well_depth: int|float=5000, 
                temperature_node: int|float=600, pressure_node: int|float=5000, angle: int|float=90, ql_i: int|float=0.001, ql_n: int|float=1000, amount: int=25):
        """
        Args:
            pressure (int | float): Well Pressure [psia]
            temperature (int | float): Well Temperature [oR]
            specific_gravity (float, optional): Gas Specific Gravity. Defaults to 0.65.
            api (int | float, optional): API Specific. Defaults to 40.
            bubble_pressure (int | float, optional): Bubble Pressure. Defaults to 0.
            salinity (int | float, optional): Salinity [ppm]. Defaults to 1000.
            water_cut (float, optional): Water Cut. Defaults to 0.0.
            go_ratio (int | float, optional): Gas-Oil Ratio [scf/stb]. Defaults to 300.
            internal_diameter (int | float, optional): Inside Pipe Diameter [in]. Defaults to 2.5.
            rugosity (float, optional): Pipe Rugosity [in]. Defaults to 0.0001.
            well_depth (int | float, optional): Well Depth [ft]. Defaults to 5000.
            temperature_node (int | float, optional): Temperature Node [oR]. Defaults to 600.
            pressure_node (int | float, optional): Pressure Node [psia]. Defaults to 5000.
            angle (int | float, optional): Well Angle. Defaults to 90.
            ql_i (int | float, optional): Initial Rate Liquid [bbl/d]. Defaults to 0.001.
            ql_n (int | float, optional): Max Rate Liquid [bbl/d]. Defaults to 1000.
            amount (int, optional): Number of Points. Defaults to 25.
        ### Private Args:
            wo_ratio (float): Water-Oil Ratio [stb/stb]
            sg_oil (float): Oil Specific Gravity
            gl_ratio (float): Gas-Liquid Ratio [scf/stb]
            area (float): Area of Tubbing [ft]
            delta_ql (list[float]): Liquid rates assumed from ql_i to ql_n [bbl/d]
            delta_depth (list[float]): Calculate depth from 0 to well_depth
            delta_t (list[float]): Calculate flowing temperature from 0 to temperature_node             
        """        
        
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
        self.pressure_node = pressure_node
        self.angle = angle
        self.ql_i = ql_i
        self.ql_n = ql_n
        self.amount = amount
        
        self.wo_ratio = self.water_cut/(1-self.water_cut)
        self.sg_oil = 141.5/(131.5+self.api)
        self.gl_ratio = (self.go_ratio)/(self.wo_ratio+1)
        self.area = np.pi/4*((self.internal_diameter/12)**2)
        self.angle_value = (self.angle*np.pi)/180
        
        self.delta_ql = np.linspace(self.ql_i, self.ql_n, self.amount)
        self.delta_depth = np.linspace(0, self.well_depth, self.amount)
        self.delta_t = self._delta_temp_()
        
        self._prop_gas = GasProperties(self.pressure, self.temperature, self.specific_gravity)
        self._prop_oil = OilProperties(self.pressure, self.temperature, self.specific_gravity, self.api, bubble_pressure=self.bubble_pressure)
        self._prop_water = WaterProperties(self.pressure, self.temperature, self.salinity)
        self._rho_liq, self._mu_liq, self._sigma_liq = self._properties_liquid_()
        self._q_water, self._q_oil, self._q_gas, self._q_liq = self._flow_()
        self._v_sl, self._v_sg, self._v_m = self._velocities_()
        self._Nfr, self._Nlv, self._l1, self._l2, self._l3, self._l4, self._lambda_liq, self._lambda_gas = self._flow_regime_()
        self._regime = self._regime_()
        self._holdup = self.holdup()
        self._rho_m, self._mu_m, self._rho_mis = self._properties_mixture_()
        self._NRe, self._ft = self._number_reynolds_()
        
    def _delta_temp_(self):
        gradient = np.abs((self.temperature_node-self.temperature))/self.well_depth
        return self.temperature + gradient*self.delta_depth
    
    def _properties_liquid_(self):
        rho_liq = self._prop_oil.density_oil()*(1/(1+self.wo_ratio)) + self._prop_water.density_water()*(self.wo_ratio/(1+self.wo_ratio))
        mu_liq = self._prop_oil.viscosity_oil()*(1/(1+self.wo_ratio)) + self._prop_water.viscosity_water()*(self.wo_ratio/(1+self.wo_ratio))
        sigma_oil = self._prop_oil.tension_oil()
        if sigma_oil < 1:
            sigma_oil = 1
        sigma_water = self._prop_water.tension_water()
        if sigma_water < 1:
            sigma_water = 1
        sigma_liq = sigma_oil*(1/(1+self.wo_ratio)) + sigma_water*(self.wo_ratio/(1+self.wo_ratio))
        return [rho_liq, mu_liq, sigma_liq]
    
    def _flow_(self):
        q_liq = self.ql_i
        q_oil = self.ql_i*(1-self.water_cut)
        q_water = self.ql_i*self.water_cut
        q_gas = self.ql_i*self.gl_ratio
        return [q_water, q_oil, q_gas, q_liq]
    
    def _velocities_(self):
        #v_sl = (((self._prop_oil.factor_volumetric_oil()*self._q_oil)/15387) + ((self._prop_water.factor_volumetric_water()*self._q_water)/15387))/self.area
        v_sl = ((5.615*self._q_liq)/(86400*self.area))*(self._prop_oil.factor_volumetric_oil()*(1/(1+self.wo_ratio)) + self._prop_water.factor_volumetric_water()*(self.wo_ratio/(1+self.wo_ratio)))
        #v_sg = ((self._prop_gas.factor_volumetric_gas()*self._q_gas)/15387)/self.area
        v_sg = ((self._q_liq*(self.gl_ratio - self._prop_oil.solution_oil()*(1/(1+self.wo_ratio))))/(86400*self.area))*self._prop_gas.factor_volumetric_gas()
        if v_sg < 0:
            v_sg = 0
        v_m = v_sl + v_sg
        return [v_sl, v_sg, v_m]
    
    def _flow_regime_(self):
        Nfr = (self._v_m**2)/((self.internal_diameter/12)*32.174)
        Nlv = 1.938*self._v_sl*((self._rho_liq/self._sigma_liq)**0.25)
        lambda_liq = self._v_sl / self._v_m
        lambda_gas = 1-lambda_liq
        l1 = 316*(lambda_liq**0.302)
        l2 = 9.252e-4*(lambda_liq**(-2.4684))
        l3 = 0.10*(lambda_liq**(-1.4516))
        l4 = 0.5*(lambda_liq**(-6.738))
        return [Nfr, Nlv, l1, l2, l3, l4, lambda_liq, lambda_gas]
    
    def _regime_(self):
        if ((self._lambda_liq<0.01 and self._Nfr<1) or (self._lambda_liq>=0.01 and self._Nfr<self._l2)):
            return 'segregated flow'
        elif (self._lambda_liq>=0.01 and self._l2<self._Nfr<=self._l3):
            return 'transition flow'
        elif ((0.01<=self._lambda_liq<0.4 and self._l3<self._Nfr<=self._l1) or (self._lambda_liq>=0.4 and self._l3<self._Nfr<=self._l4)):
            return 'intermittent flow'
        elif ((self._lambda_liq<0.4 and self._Nfr>=self._l1) or (self._lambda_liq>=0.4 and self._Nfr>self._l4)):
            return 'distributed flow'
    
    def holdup(self):
        """
        Returns:
            Array: It will returns an array value of the holdup.
        """
        
        def hold_l(nfr_v, nlv_v, lambl_v, regime_v, l2_v, l3_v):
            if regime_v == 'segregated flow':
                a, b, c = 0.98, 0.4846, 0.0868
                if self.angle_value >= 0:
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
                if self.angle_value >= 0:
                    d, e, f, g = 0.011, -3.768, 3.539, -1.614
                else:
                    d, e, f, g = 4.7, -0.3692, 0.1244, -0.5056
            elif regime_v == 'distributed flow':
                a, b, c = 1.065, 0.5824, 0.0609
                if self.angle_value >= 0:
                    d, e, f, g = 1, 0, 0, 0
                else:
                    d, e, f, g = 4.7, -0.3692, 0.1244, -0.5056
                    
            cof_incl = (1-lambl_v)*np.log(d*(lambl_v**e)*(nlv_v**f)*(nfr_v**g))
            
            if cof_incl < 0:
                cof_incl = 0
                
            psi = 1 + cof_incl*(np.sin(1.8*self.angle_value) - 0.333*(np.sin(1.8*self.angle_value)**3))
            
            hold_liq_h = (a*(lambl_v**b))/(nfr_v**c)
                
            if hold_liq_h < lambl_v:
                hold_liq_h = lambl_v
            
            holdup_liq = hold_liq_h*psi
            
            if holdup_liq > 1.0:
                holdup_liq = 1.0
            
            return holdup_liq
        
        holdup_liquid = hold_l(self._Nfr, self._Nlv, self._lambda_liq, self._regime, self._l2, self._l3)
        
        return holdup_liquid  
    
    def _properties_mixture_(self):
        rho_m = self._rho_liq*self._lambda_liq + self._prop_gas.density_gas()*self._lambda_gas
        mu_m = self._mu_liq*self._lambda_liq + self._prop_gas.viscosity_gas()*self._lambda_gas
        rho_mis = self._rho_liq*self._holdup + self._prop_gas.density_gas()*(1-self._holdup)
        return [rho_m, mu_m, rho_mis]
    
    def _number_reynolds_(self):
        NRe = (1488*self._rho_m*self._v_m*(self.internal_diameter/12))/(self._mu_m)
        fn = (-2*np.log10((1/3.7)*(self.rugosity/self.internal_diameter))+((6.81/NRe)**0.9))**(-2)
        y = self._lambda_liq/(self._holdup**2)
        
        if (y>1 and y<1.2):
            s = np.log(2.2*y - 1.2)
        else:
            s = np.log(y)/((-0.0523) + (3.182*np.log(y)) - (0.8725*((np.log(y))**2)) + (0.01853*((np.log(y))**4)))
        
        ft = fn*np.exp(s)
        
        return [NRe, ft]

    def pressure_traverse(self):
        """
        Returns:
            List[array]: It will return a list from the ql_i argument with the values of liquid velocity, gas velocity, mix velocity, drop pressure, flowing pressure\n
            [vsl (ft/s), vsg (ft/s), vm (ft/s), dp (psia), p (psia)]
        """
        p = [self.pressure]
        vsl = [self._v_sl]
        vsg = [self._v_sg]
        vm = [self._v_m]
        hl = [self._holdup]
        dpf_i = (self._ft*self._rho_mis*(self._v_m**2))/(2*32.17*(self.internal_diameter/12))
        dph_i = (32.17*self._rho_mis*np.sin(self.angle_value))/(32.17)
        ek_i = (self._rho_mis*self._v_m*self._v_sg)/(144*32.17*self.pressure)
        dpt_i = (dpf_i + dph_i)/(144*(1-ek_i))
        dp = [dpt_i]
        
        for dh, ti, i in zip(self.delta_depth, self.delta_t, range(self.amount)):
            
            if i == 0:
                continue
            
            pi = p[i-1]+dp[i-1]*(dh-self.delta_depth[i-1])
            self._prop_gas = GasProperties(pi, ti, self.specific_gravity)
            self._prop_oil = OilProperties(pi, ti, self.specific_gravity, self.api, bubble_pressure=self.bubble_pressure)
            self._prop_water = WaterProperties(pi, ti, self.salinity)
            self._rho_liq, self._mu_liq, self._sigma_liq = self._properties_liquid_()
            self._q_water, self._q_oil, self._q_gas, self._q_liq = self._flow_()
            self._v_sl, self._v_sg, self._v_m = self._velocities_()
            self._Nfr, self._Nlv, self._l1, self._l2, self._l3, self._l4, self._lambda_liq, self._lambda_gas = self._flow_regime_()
            self._regime = self._regime_()
            self._holdup = self.holdup()
            self._rho_m, self._mu_m, self._rho_mis = self._properties_mixture_()
            self._NRe, self._ft = self._number_reynolds_()
            
            dpf = (self._ft*self._rho_mis*(self._v_m**2))/(2*32.17*(self.internal_diameter/12))
            dph = (32.17*self._rho_mis*np.sin(self.angle_value))/(32.17)
            ek = (self._rho_mis*self._v_m*self._v_sg)/(144*32.17*pi)
            
            dpt = (dpf + dph)/(144*(1-ek))
            pj = p[i-1]+(dpt)*(dh-self.delta_depth[i-1])
            
            vsl.append(self._v_sl)
            vsg.append(self._v_sg)
            vm.append(self._v_m)
            hl.append(self._holdup)
            
            dp.append(dpt)
            
            p.append(pj)
            
        return [np.array(vsl), np.array(vsg), np.array(vm), np.array(hl), np.array(dp), np.array(p)]
    
    def outflow(self):  
        """
        Returns:
            List[array]: It will returns a list with the values of liquid rate, gas rate, oil rate, water rate, flowing pressure\n
            [qli (bbl/d), qgi (Mscf/d), qoi (bbl/d), qwi (bbl/d), pwfi (psia)]
        """  
        qoi = []
        qwi = []
        qgi = []
        qli = []
        pwfi = []
    
        for flow_value in self.delta_ql:
            self.ql_i = flow_value
            self._q_water, self._q_oil, self._q_gas, self._q_liq = self._flow_()
            *_, pwf = self.pressure_traverse()
            
            if pwf[-1] > 2*self.pressure_node:
                break
            
            pwfi.append(pwf[-1])
            qoi.append(self._q_oil)
            qwi.append(self._q_water)
            qgi.append(self._q_gas/1000)
            qli.append(self._q_liq)
        
        return [np.array(qli), np.array(qgi), np.array(qoi), np.array(qwi), np.array(pwfi)]
        
            
if __name__ == "__main__":
    # import time
    # time_start = time.time()
    well = BeggsBrill(450, (100+460))
    #well2 = BeggsBrill(350, (100+460), bubble_pressure=1500, internal_diameter=1.5)
    #well3 = BeggsBrill(450, (100+460), bubble_pressure=1500, water_cut=0.6)
    
    #print(well.pressure_traverse())
    
    #print(well.outflow())
    #import matplotlib.pyplot as plt
    
    # ql1, qg1, qo1, qw1, pw1 = well1.outflow()
    # ql2, qg2, qo2, qw2, pw2 = well2.outflow()
    # ql3, qg3, qo3, qw3, pw3 = well3.outflow()
    # print('Well1')
    # print(ql1, pw1)
    # print('Well2')
    # print(ql2, pw2)
    # print('Well3')
    # print(ql3, pw3)
    # plt.plot(ql1, pw1)
    # plt.plot(ql2, pw2)
    # plt.plot(ql3, pw3)
    
    # h = well.delta_depth
    # vl, vg, vm, hl, dp, p = well.pressure_traverse()
    # print("vl", vl, "vg", vg, "vm", vm, "hl", hl, "dp", dp, "P", p)
    # fig, ax = plt.subplots(2, 3)
    
    # ax[0, 0].invert_yaxis()
    # ax[0, 0].plot(p, h)
    
    # ax[0, 1].invert_yaxis()
    # ax[0, 1].plot(dp, h)
    
    # ax[0, 2].invert_yaxis()
    # ax[0, 2].plot(hl, h)
    
    # ax[1, 0].invert_yaxis()
    # ax[1, 0].plot(vl, h)
    
    # ax[1, 1].invert_yaxis()
    # ax[1, 1].plot(vg, h)
    
    # ax[1, 2].invert_yaxis()
    # ax[1, 2].plot(vm, h)
    
    # time_end = time.time()
    # print('time', time_end-time_start)
    # plt.show()
    