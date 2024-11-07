import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
import numpy as np
from _properties.gasProperties import GasProperties

class LITRD:
    def __init__(self, pressure: int|float, temperature: int|float, bubble_pressure: int|float=0, specific_gravity: float=0.65, api: int|float=40,
                permeability: int|float=10, compressibility: float=1e-5, skin: int|float=0, height_formation: int|float=10, 
                well_radius: int|float=0.35, reservoir_radius: int|float=1000, water_cut: float=0.0, 
                go_ratio: int|float=5000, amount: int=25):
        
        self.pressure = pressure
        self.temperature = temperature
        self.bubble_pressure = bubble_pressure
        self.specific_gravity = specific_gravity
        self.api = api
        self.permeability = permeability
        self.compressibility = compressibility
        self.skin = skin
        self.height_formation = height_formation
        self.well_radius = well_radius
        self.reservoir_radius = reservoir_radius
        self.water_cut = water_cut
        self.go_ratio = go_ratio
        self.amount = amount
        
        self.delta_p = np.linspace(self.pressure, 14.7, self.amount)
        
        self._mp = self._potential_()
        
    def _ratio_(self):
        prop_gas = GasProperties(self.delta_p, self.temperature, self.specific_gravity)
        z = prop_gas.factor_compressibility_gas()
        mu_g = prop_gas.viscosity_gas()
        return self.delta_p/(mu_g*z)
    
    def _potential_(self):
        ratio = np.flip(self._ratio_())
        pressure = np.flip(self.delta_p)
        
        delta_mp = []
        
        for i, r in enumerate(ratio):
            if i == 0:
                mp = ((0+r)*(pressure[i]-0)) + 0
                delta_mp.append(mp)
            else:
                mp = ((ratio[i-1]+r)*(pressure[i]-pressure[i-1])) + delta_mp[i-1]
                delta_mp.append(mp)
        
        delta_mp = np.array(delta_mp)
        
        return np.flip(delta_mp)
    
    def _flow_gas_(self):
        mpi = self._mp[0]
        return (703e-6*self.permeability*self.height_formation*(mpi-self._mp))/(self.temperature*(np.log(self.reservoir_radius/self.well_radius)-0.75+self.skin))
    
    def inflow(self):
        qg = self._flow_gas_()*1000
        qo = qg/self.go_ratio
        qw = (qg*self.water_cut*(1/self.go_ratio))/(1-self.water_cut)
        ql = qo + qw
        return [ql, qg/1000, qo, qw, self.delta_p]
    
class LITPD:
    def __init__(self, pressure: int|float, temperature: int|float, qg_test: List[int|float]=[3000, 4500, 5600, 6300], pwf_test: List[int|float]=[3500, 3000, 2500, 2000], 
                specific_gravity: float=0.65, water_cut: float=0.0, go_ratio: int|float=5000, amount: int=25):
        
        self.pressure = pressure
        self.temperature = temperature
        self.qg_test = np.array(qg_test)/1e3
        self.pwf_test = np.array(pwf_test)
        self.specific_gravity = specific_gravity
        self.water_cut = water_cut
        self.go_ratio = go_ratio
        self.amount = amount
        
        self.delta_p = np.linspace(self.pressure, 14.7, self.amount)
        
        self._mp = self._potential_()
        
        self._a, self._b = self._linear_regression_()
        
    def _ratio_(self):
        prop_gas = GasProperties(self.delta_p, self.temperature, self.specific_gravity)
        z = prop_gas.factor_compressibility_gas()
        mu_g = prop_gas.viscosity_gas()
        return self.delta_p/(mu_g*z)
        
    def _potential_(self):
        ratio = np.flip(self._ratio_())
        pressure = np.flip(self.delta_p)
        
        delta_mp = []
        
        for i, r in enumerate(ratio):
            if i == 0:
                mp = ((0+r)*(pressure[i]-0))/1e6 + 0
                delta_mp.append(mp)
            else:
                mp = ((ratio[i-1]+r)*(pressure[i]-pressure[i-1]))/1e6 + delta_mp[i-1]
                delta_mp.append(mp)
        
        delta_mp = np.array(delta_mp)
        
        return np.flip(delta_mp)
    
    def _quadratic_regression_(self):
        n = len(self.delta_p)
        p = np.sum(self.delta_p)
        mp = np.sum(self._mp)
        p2 = np.sum(self.delta_p**2)
        p3 = np.sum(self.delta_p**3)
        p4 = np.sum(self.delta_p**4)
        p_mp = np.sum(self._mp*self.delta_p)
        p2_mp = np.sum((self.delta_p**2)*self._mp)
        
        A = np.array([[p2, p, n],
                    [p3, p2, p],
                    [p4, p3, p2]])
        
        B = np.array([mp, p_mp, p2_mp])
        
        a, b, c = np.linalg.solve(A, B)
        
        return a, b, c
        
    def _linear_regression_(self):
        
        mpi = self._mp[0]
        a, b, c = self._quadratic_regression_()
        mp = a*(self.pwf_test**2) + b*(self.pwf_test) + c
        delta_mp = mpi - mp 
        d_q = delta_mp/self.qg_test #y
        q_dq = self.qg_test*d_q #x*y
        q2 = self.qg_test**2 #x^2        
        
        a = (len(self.pwf_test)*np.sum(q_dq) - np.sum(self.qg_test)*np.sum(d_q))/(len(self.pwf_test)*np.sum(q2)-(np.sum(self.qg_test))**2)
        b = (np.sum(d_q)*np.sum(q2) - np.sum(self.qg_test)*np.sum(q_dq))/(len(self.pwf_test)*np.sum(q2)-(np.sum(self.qg_test))**2)
        
        return a, b
    
    def _flow_gas_(self):
        mpi = self._mp[0]
        return (-np.abs(self._b) + np.sqrt(np.abs(self._b)**2 + 4*np.abs(self._a)*(mpi-self._mp)))/(2**np.abs(self._a))
    
    def inflow(self):
        qg = self._flow_gas_()*1e6
        qo = qg/self.go_ratio
        qw = (qg*self.water_cut*(1/self.go_ratio))/(1-self.water_cut)
        ql = qo + qw
        return [ql, qg/1e3, qo, qw, self.delta_p]
    
if __name__ == '__main__':
    #well = LITRD(6100, 857)
    well = LITPD(6100, 857,)# qg_test=[6000, 9000, 14000, 17000], pwf_test=[7055, 6678, 6123, 5756])
    #print(well._linear_regression_())
    
    # ql, qg, qo, qw, pw = well.inflow()
    # print(ql, qg, qo, qw, pw)
    
    # import matplotlib.pyplot as plt
    # q = well._flow_gas_()
    # p = well.delta_p
    # print(q, p)
    # plt.plot(q, p)
    # plt.show()
    
    # fig, (mp, mpc) = plt.subplots(1, 2)
    # p = well.delta_p
    # m = well._mp
    # mc = well._mpc
    
    # mp.plot(m, p)
    # mpc.plot(mc, p)
    # plt.show()
    
    # fig, axs = plt.subplots(2, 2)
    # #print('fig', fig, 'axs', axs)
    # axs[0, 0].plot(ql, pw)
    # axs[0, 1].plot(qg, pw)
    # axs[1, 0].plot(qo, pw)
    # axs[1, 1].plot(qw, pw)
    # #plt.plot(qo, pw)
    # plt.show()