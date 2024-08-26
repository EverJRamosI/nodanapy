import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.optimize import root
from _properties.gasProperties import GasProperties
from _properties.oilProperties import OilProperties
from _properties.waterProperties import WaterProperties

class HagedornBrown:
    def __init__(self, pressure: int|float, temperature: int|float, specific_gravity: float=0.65, 
                api: int|float=40, bubble_pressure: int|float=0, salinity: int|float=1000, water_cut: float=0.1, 
                go_ratio: int|float=50, internal_diameter: int|float=2.5, rugosity: float=0.0001, well_depth: int|float=500, 
                qg_max: int|float=10000, amount: int=25):
        
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
        self.qg_max = qg_max
        self.amount = amount
        
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
        rs_oil = properties.solution_oil()
        fvf_oil = properties.factor_volumetric_oil()
        return [mu_oil, rho_oil, sigma_oil, rs_oil, fvf_oil]
    
    def _properties_water_(self):
        properties = WaterProperties(self.pressure, self.temperature, self.salinity)
        mu_water = properties.viscosity_water()
        rho_water = properties.density_water()
        sigma_water = properties.tension_water()
        fvf_water = properties.factor_volumetric_water()
        return [mu_water, rho_water, sigma_water, fvf_water]
    
    def _flow_(self):
        q_lm = ((self.go_ratio*(self.qg_max*1.2))/(1-self.water_cut))/1000
        q_liq = np.linspace(1, q_lm, self.amount)
        q_oil = (1-self.water_cut)*q_liq
        q_water = q_liq-q_oil
        lgr = self.go_ratio/(1-self.water_cut)
        q_gas = (q_liq*1000)/lgr
        return [q_water, q_oil, q_gas, q_liq]
    
    def _fractions_liquid_(self):
        flow_values = self._flow_()
        q_oil = flow_values[1]
        q_water = flow_values[0]
        f_oil = q_oil/(q_oil+q_water)
        f_water = 1-f_oil
        return [f_oil, f_water]
    
    # def _delta_depth_(self):
    #     return np.linspace(0, self.well_depth, self.amount)
    
    # def _delta_pressure_(self):
    #     return self.pressure + self.pressure*0.5
    
    def _velocities_(self):
        b_g = self._properties_gas_()[2] #[3]
        bo = self._properties_oil_()[4]
        bw = self._properties_water_()[3]
        rs_o = self._properties_oil_()[3]
        fo, fw = self._fractions_liquid_()
        flow_values = self._flow_()
        q_liq = flow_values[3]
        glr = ((1-self.water_cut)/self.go_ratio)*1_000_000
        v_sg = ((q_liq*(glr-rs_o*fo))/(((np.pi/4)*((self.internal_diameter/12)**2))*86400))*b_g #*(14.7/self.pressure)*(self.temperature/520)*(z_g)
        v_sl = ((5.615*(q_liq))/((np.pi/4)*((self.internal_diameter/12)**2)*86400))*(bo*fo+bw*fw)
        v_m = v_sg + v_sl
        return [v_sg, v_sl, v_m]
    
    def _densities_(self):
        rs_o = self._properties_oil_()[3]
        bo = self._properties_oil_()[4]
        rho_o = self._properties_oil_()[1]
        rho_w = self._properties_water_()[1]
        rho_g = self._properties_gas_()[1]
        rho_liq = ((rho_o+((rs_o*0.0764*self.specific_gravity)/(5.614)))/bo)*self._fractions_liquid_()[0]+rho_w*self._fractions_liquid_()[1]
        return [rho_liq, rho_g]
    
    def _sigma_tensions_(self):
        fo, fw = self._fractions_liquid_()
        sigma_gw = self._properties_water_()[2]
        sigma_go = self._properties_oil_()[2]
        sigma_liq = sigma_go*fo + sigma_gw*fw
        return [sigma_gw, sigma_go, sigma_liq]
    
    def _viscosities_(self):
        mu_water = self._properties_water_()[0]
        mu_oil = self._properties_oil_()[0]
        mu_gas = self._properties_gas_()[0]
        mu_liq = mu_water*self._fractions_liquid_()[1] + mu_oil*self._fractions_liquid_()[0]
        return [mu_water, mu_oil, mu_gas, mu_liq]
    
    def _holdup_(self):
        velocities = self._velocities_()
        vsg = velocities[0]
        vsl = velocities[1]
        o_liq = self._sigma_tensions_()[2]
        rho_l = self._densities_()[0]
        mu_l = self._viscosities_()[3]
        NLV = 1.938*vsl*((rho_l/o_liq)**(1/4))
        NGV = 1.938*vsg*((rho_l/o_liq)**(1/4))
        ND = 120.872*(self.internal_diameter/12)*((rho_l/o_liq)**(1/2))
        NL = 0.15726*mu_l*((1/(rho_l*(o_liq**3)))**(1/4))
        CNL = 0.061*(NL**3)-0.0929*(NL**2)+0.0505*(NL)+0.0019
        H = (NLV/(NGV**0.575))*((self.pressure/14.7)**0.1)*(CNL/ND)
        holdup_psi = np.sqrt((0.0047+1123.32*H+729489.64*(H**2))/(1+1097.1566*H+722153.97*(H**2)))
        B = (NGV*(NLV**0.38))/(ND**2.14)
        
        
        # psi = []
        # for b in B:
        #     if b <= 0.025:
        #         p = 27170*(b**3)-314.52*(b**2)+0.5472*b+0.9999
        #     elif b > 0.025:
        #         p = -533.33*(b**2)+58.524*b+0.1171
        #     elif b > 0.055:
        #         p = 2.5714*b+1.5962
        #     psi.append(p)
            
        
        psi = [
            27170*(b**3)-314.52*(b**2)+0.5472*b+0.9999 if b <= 0.025 else
            (-533.33*(b**2)+58.524*b+0.1171 if b > 0.025 and b <= 0.055 else
            2.5714*b+1.5962)
            for b in B
        ]

        
            
        return holdup_psi*psi
    
    def _rho_m_(self):
        rho_l, rho_g = self._densities_()
        return rho_l*self._holdup_()+rho_g*(1-self._holdup_())
                    
    def pressure_drop_gravity(self):
        vm = self._velocities_()[2]
        delta_pressure = self._rho_m_()*(((vm**2)/(2*32.17))/(144*32.17))
        return delta_pressure
    
    def _number_reynolds_(self):
        q_liq = self._flow_()[3]
        rho_w = self._properties_water_()[1]
        mu_l = self._viscosities_()[3]
        mu_g = self._viscosities_()[2]
        Hl = self._holdup_()
        fo, fw = self._fractions_liquid_()
        glr = ((1-self.water_cut)/self.go_ratio)*1_000
        sg_oil = 141.5/(131.5+self.api)
        M = sg_oil*350.52*fo + (rho_w/62.42)*350.52*fw + self.specific_gravity*0.0764*glr
        #print(glr)
        #print(M)
        NRe = 2.2e-2*((q_liq*M)/((self.internal_diameter/12)*(mu_l**Hl)*(mu_g**(1-Hl))))
        #NRe = (self._densities_()[2]*(self.internal_diameter/12))/(0.000672*self._viscosities_()[4])
        f = (-2*np.log10((1/3.7)*(self.rugosity/self.internal_diameter))+((6.81/NRe)**0.9))**(-2)
        #def f_cal(f, NRe):
        #    return 1/np.sqrt(f)+2*np.log10((self.rugosity/(3.7*self.internal_diameter))+(2.51/(NRe/np.sqrt(f))))
        
        #f = [root(f_cal, 0.01, args=(nre, )).x[0] for nre in NRe]
        
        
        
        return [NRe, f]
    
    def pressure_drop_friction(self):
        q_liq = self._flow_()[3]
        friction = self._number_reynolds_()[1]
        rho_w = self._properties_water_()[1]
        fo, fw = self._fractions_liquid_()
        sg_oil = 141.5/(131.5+self.api)
        glr = ((1-self.water_cut)/self.go_ratio)*1_000
        M = sg_oil*350.52*fo + (rho_w/62.42)*350.52*fw + self.specific_gravity*0.0764*glr
        delta_pressure = self.well_depth*(friction*(M**2)*(q_liq**2))/(144*2.9652e11*((self.internal_diameter/12)**5)*self._rho_m_())
        return delta_pressure
    
    def bottom_hole_pressure(self):
        dp_gravity = self.pressure_drop_gravity()
        dp_friction = self.pressure_drop_friction()
        dp_kinetic = self._rho_m_()
        dp_total = dp_gravity + dp_friction + dp_kinetic
        pwf = self.pressure + dp_total
        return [dp_total, pwf]
    
    def outflow(self):
        qw, qo, qg, ql = self._flow_()
        pwf = self.bottom_hole_pressure()[1]
        return [ql, qg/1000, qo, qw, pwf]
    
    
if __name__ == "__main__":
    well = HagedornBrown(845, 650, well_depth=8034.77)
    
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
    
    qg = well.outflow()[1]
    pwf = well.outflow()[4]
    print("Qg", qg)
    print("Pwf", pwf)
    
    import matplotlib.pyplot as plt
    
    plt.plot(qg, pwf)
    plt.show()