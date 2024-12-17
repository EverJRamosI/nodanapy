import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties
from _properties.oilProperties import OilProperties
from _properties.waterProperties import WaterProperties

class HagedornBrown:
    """
    ### Summary:
    This class is to determine VLP with the Hagerdorn Brown method.
    
    ### Methods:
    - __delta_temp_
    - __flow_
    - __properties_liquid_
    - __total_mass_
    - __velocities_
    - holdup: This method determines the holdup
    - __rho_m_
    - __number_reynolds_
    - pressure_traverse: This method determines the flowing pressure
    - outflow: This method is to calculate the VLP.
    """    
    def __init__(self, pressure: int|float, temperature: int|float, specific_gravity: float=0.65, 
                api: int|float=40, bubble_pressure: int|float=0, salinity: int|float=1000, water_cut: float=0.0, 
                go_ratio: int|float=500, internal_diameter: int|float=2.5, rugosity: float=0.0001, well_depth: int|float=5000, 
                temperature_node: int|float=600, pressure_node: int|float=5000, ql_i: int|float=0.001, ql_n: int|float=1000, amount: int=25):
        """
        Args:
            pressure (int | float): Well Pressure [psia]
            temperature (int | float): Well Temperature [oR]
            specific_gravity (float, optional): Gas Specific Gravity. Defaults to 0.65.
            api (int | float, optional): API Specific. Defaults to 40.
            bubble_pressure (int | float, optional): Bubble Pressure. Defaults to 0.
            salinity (int | float, optional): Salinity [ppm]. Defaults to 1000.
            water_cut (float, optional): Water Cut. Defaults to 0.0.
            go_ratio (int | float, optional): Gas-Oil Ratio [scf/stb]. Defaults to 500.
            internal_diameter (int | float, optional): Inside Pipe Diameter [in]. Defaults to 2.5.
            rugosity (float, optional): Pipe Rugosity [in]. Defaults to 0.0001.
            well_depth (int | float, optional): Well Depth [ft]. Defaults to 5000.
            temperature_node (int | float, optional): Temperature Node [oR]. Defaults to 600.
            pressure_node (int | float, optional): Pressure Node [psia]. Defaults to 5000.
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
        self.ql_i = ql_i
        self.ql_n = ql_n
        self.amount = amount
        
        self.wo_ratio = self.water_cut/(1-self.water_cut)
        self.sg_oil = 141.5/(131.5+self.api)
        self.gl_ratio = (self.go_ratio)/(self.wo_ratio+1)
        self.area = np.pi/4*((self.internal_diameter/12)**2)
        
        self.delta_ql = np.linspace(self.ql_i, self.ql_n, self.amount)
        self.delta_depth = np.linspace(0, self.well_depth, self.amount)
        self.delta_t = self._delta_temp_()
        
        self._q_water, self._q_oil, self._q_gas, self._q_liq = self._flow_()
        self._prop_gas = GasProperties(self.pressure, self.temperature, self.specific_gravity)
        self._prop_oil = OilProperties(self.pressure, self.temperature, self.specific_gravity, self.api, bubble_pressure=self.bubble_pressure)
        self._prop_water = WaterProperties(self.pressure, self.temperature, self.salinity)
        self._rho_liq, self._mu_liq, self._sigma_liq = self._properties_liquid_()
        self._mass = self._total_mass_()
        self._v_sl, self._v_sg, self._v_m = self._velocities_()
        self._holdup = self.holdup()
        self._rho_m = self._rho_m_()
        self._NRe, self._f = self._number_reynolds_()
    
    def _delta_temp_(self):
        gradient = np.abs((self.temperature_node-self.temperature))/self.well_depth
        return self.temperature + gradient*self.delta_depth
            
    def _flow_(self):
        q_oil = (1-self.water_cut)*self.ql_i
        q_gas = self.ql_i*self.gl_ratio
        q_water = self.water_cut*self.ql_i
        return [q_water, q_oil, q_gas, self.ql_i]
    
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
    
    def _total_mass_(self):
        return self.sg_oil*350.52*(1/(1+self.wo_ratio)) + (self._prop_water.density_water()/62.42)*350.52*(self.wo_ratio/(1+self.wo_ratio)) + self.specific_gravity*0.0764*self.gl_ratio 
    
    def _velocities_(self):
        v_sl = ((5.615*self._q_liq)/(86400*self.area))*(self._prop_oil.factor_volumetric_oil()*(1/(1+self.wo_ratio)) + self._prop_water.factor_volumetric_water()*(self.wo_ratio/(1+self.wo_ratio)))
        v_sg = ((self._q_liq*(self.gl_ratio - self._prop_oil.solution_oil()*(1/(1+self.wo_ratio))))/(86400*self.area))*self._prop_gas.factor_volumetric_gas()
        if v_sg < 0:
            v_sg = 0
        v_m = v_sl + v_sg
        return [v_sl, v_sg, v_m]
    
    def holdup(self):
        """
        Returns:
            Array: It will returns an array value of the holdup.
        """ 
        
        A = 1.071 - ((0.2218*((self._v_m)**2))/(self.internal_diameter/12))
        
        if A < 0.13:
            A = 0.13
        
        B = self._v_sg/self._v_m
        
        if B > A:
            NLV = 1.938*self._v_sl*((self._rho_liq/self._sigma_liq)**(1/4))
            NGV = 1.938*self._v_sg*((self._rho_liq/self._sigma_liq)**(1/4))
            ND = 120.872*(self.internal_diameter/12)*(np.sqrt(self._rho_liq/self._sigma_liq))
            NL = 0.15726*self._mu_liq*((1/(self._rho_liq*(self._sigma_liq**3)))**(1/4))
            X1 = np.log10(NL) + 3
            Y = -2.69851 + (0.51841*X1) - (0.551*(X1**2)) + (0.54785*(X1**3)) - (0.12195*(X1**4))
            CNL = 10**Y
            
            X2 = (NLV*(self.pressure**0.1)*CNL)/((NGV**0.575)*(14.7**0.1)*ND)
            
            holdup_psi = -0.10307 + 0.61777*(np.log10(X2)+6) - 0.63295*((np.log10(X2)+6)**2) + 0.29598*((np.log10(X2)+6)**3) - 0.0401*((np.log10(X2)+6)**4)
            
            X3 = (NGV*(NL**0.38))/(ND**2.14)
            
            if X3 < 0.01:
                X3 = 0.01
            
            psi = 0.91163 - 4.82176*X3 + 1232.25*(X3**2) - 22253.6*(X3**3) + 116174.3*(X3**4)
            
            holdup_liq = holdup_psi*psi
    
        else:
            vs = 0.8
            holdup_liq = 1 - 0.5*(1 + (self._v_m/vs) - np.sqrt(((1 + (self._v_m/vs))**2) - (4*(self._v_sg/vs))))
        
        if holdup_liq > 1.0:
            holdup_liq = 1.0
        
        return holdup_liq
    
    def _rho_m_(self):
        return self._rho_liq*self._holdup+self._prop_gas.density_gas()*(1-self._holdup)
    
    def _number_reynolds_(self):
        q_liq = ((self._prop_oil.factor_volumetric_oil()*self._q_oil)/15387) + ((self._prop_water.factor_volumetric_water()*self._q_water)/15387)
        
        NRe = 2.2e-2*((q_liq*15387*self._mass)/((self.internal_diameter/12)*(self._mu_liq**self._holdup)*(self._prop_gas.viscosity_gas()**(1-self._holdup))))
        if NRe < 4000:
            f = (-2*np.log10((1/3.7)*(self.rugosity/self.internal_diameter))+((6.81/NRe)**0.9))**(-2)
        else:
            f = 1/(-4*np.log10((self.rugosity/3.7065)-(5.0452/NRe*(np.log10((self.rugosity**(1.1098)/2.8257)+((7.149/NRe)**0.8991))))))**2
        return [NRe, f]
        
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
        dpf_i = (self._rho_m + (self._f*(self._mass**2)*((self._q_liq)**2))/(7.413e10*((self.internal_diameter/12)**5)*self._rho_m))/144
        dp = [dpf_i]
        
        for dh, ti, i in zip(self.delta_depth, self.delta_t, range(self.amount)):
            
            if i == 0:
                continue
            
            pi = p[i-1]+dp[i-1]*(dh-self.delta_depth[i-1])
                
            self._prop_gas = GasProperties(pi, ti, self.specific_gravity)
            self._prop_oil = OilProperties(pi, ti, self.specific_gravity, self.api, bubble_pressure=self.bubble_pressure)
            self._prop_water = WaterProperties(pi, ti, self.salinity)
            self._rho_liq, self._mu_liq, self._sigma_liq = self._properties_liquid_()
            self._mass = self._total_mass_()
            self._v_sl, self._v_sg, self._v_m = self._velocities_()
            self._holdup = self.holdup()
            self._NRe, self._f = self._number_reynolds_()
            self._rho_m = self._rho_m_()

            dpf = (self._f*(self._mass**2)*((self._q_liq/15387)**2))/(7.413e10*((self.internal_diameter/12)**5)*self._rho_m)
            dph = (self._rho_m*(((self._v_m**2)/(2*32.17))/(dh-self.delta_depth[i-1])))

            dpt = (self._rho_m + dpf + dph)/144
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
    well1 = HagedornBrown(249.7, (184+460), well_depth=8000, bubble_pressure=3000, temperature_node=(150+460), )
    # well2 = HagedornBrown(149.7, (84+460), well_depth=8000, bubble_pressure=1000, temperature_node=(150+460), internal_diameter=1.5,)
    # well3 = HagedornBrown(149.7, (84+460), go_ratio=200, well_depth=8000, bubble_pressure=1000, temperature_node=(150+460), internal_diameter=4.5, )
    # well4 = HagedornBrown(149.7, (84+460), go_ratio=1000, well_depth=8000, bubble_pressure=1000, temperature_node=(150+460), internal_diameter=5.5, )
    #print(well._flow_())
    # h = well.delta_depth
    # vl, vg, vm, hl, dp, p = well.pressure_traverse()
    # #vl, vg, vm = v
    # print(h, vl, vg, vm, hl, dp, p)
    #print(well.pressure_traverse_new())
    #print(well.outflow())
    #print(well._velocities_())
    #
    # import matplotlib.pyplot as plt
    
    # ql1, qg1, qo1, qw1, pw1 = well1.outflow()
    # ql2, qg2, qo2, qw2, pw2 = well2.outflow()
    # ql3, qg3, qo3, qw3, pw3 = well3.outflow()
    # ql4, qg4, qo4, qw4, pw4 = well4.outflow()
    # print('Pwf1', pw1)
    # print('Pwf2', pw2)
    # print('Pwf3', pw3)
    # print('Pwf4', pw4)
    # time_end = time.time()
    # print('Time', time_end - time_start)
    # plt.plot(ql1, pw1)
    # plt.plot(ql2, pw2)
    # plt.plot(ql3, pw3)
    # plt.plot(ql4, pw4)
    # plt.show()
    
    # h = well3.delta_depth
    # vl, vg, vm, hl, dp, p = well3.pressure_traverse()
    # print(h, p, dp, hl)
    
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
    