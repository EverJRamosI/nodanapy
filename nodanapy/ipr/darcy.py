import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from _properties.gasProperties import GasProperties

class Darcy:
    """
    ### Summary:
    This class is to determine IPR with the Darcy method for gas well with reservoir data.
    
    ### Methods:
    - __a_flow_
    - __b_flow_
    - __flow_gas_
    - inflow: This method is to calculate the IPR.
    """    
    def __init__(self, pressure: int|float, temperature: int|float, specific_gravity: float=0.65, 
                permeability: int|float=10, skin: int|float=0, height_formation: int|float=10, 
                well_radius: int|float=0.35, reservoir_radius: int|float=1000, water_cut: float=0.0, 
                go_ratio: int|float=50, amount: int=25):
        """
        Args:
            pressure (int | float): Well Pressure [psia]
            temperature (int | float): Well Temperature [oR]
            specific_gravity (float): Gas Specific Gravity. Defaults to 0.65.
            permeability (int | float): Permeability [md]. Defaults to 10.
            skin (int | float): Skin Factor. Defaults to 0.
            height_formation (int | float): Height Formation [ft]. Defaults to 10.
            well_radius (int | float): Well Radius [ft]. Defaults to 0.35.
            reservoir_radius (int | float): Reservoir Radius [ft]. Defaults to 1000.
            water_cut (float): Water Cut. Defaults to 0.0.
            go_ratio (int | float): Gas-Oil Ratio [scf/stb]. Defaults to 50.
            amount (int, optional): Number of Points. Defaults to 25.
        ### Private Args:
            delta_p (array): Pressures Assumed from 14.7 to well pressure [psia]
            _prop_gas (float): Properties of Gas
        """   
        
        self.pressure = pressure
        self.temperature = temperature
        self.specific_gravity = specific_gravity
        self.permeability = permeability
        self.skin = skin
        self.height_formation = height_formation
        self.well_radius = well_radius
        self.reservoir_radius = reservoir_radius
        self.water_cut = water_cut
        self.go_ratio = go_ratio
        self.amount = amount
        
        self.delta_p = np.linspace(self.pressure, 14.7, self.amount)
        
        self._prop_gas = GasProperties(self.pressure, self.temperature, self.specific_gravity)
        
    def _a_flow_(self):
        return ((1424*self._prop_gas.viscosity_gas()*self._prop_gas.factor_compressibility_gas()*self.temperature)/(self.permeability*self.height_formation))*(np.log(0.472*(self.reservoir_radius/self.well_radius))+self.skin)
        
    def _b_flow_(self):        
        beta = (2.33*1e10/(self.permeability**1.201))
        return (3.16e-12*beta*self.specific_gravity*self._prop_gas.factor_compressibility_gas()*self.temperature)/((self.height_formation**2)*self.well_radius)
        
    def _flow_gas_(self):
        self._prop_gas = GasProperties(self.delta_p, self.temperature, self.specific_gravity)
        A = self._a_flow_()
        B = self._b_flow_()
        return ((-A+np.sqrt((A**2)+(4*B*((self.pressure**2)-(self.delta_p**2)))))/(2*B))
    
    def inflow(self):
        """
        Returns:
            List: It will return a list with the rates and flowing bottom hole pressure\n
            [ql (bbl/d), qg(Mscf/d), qo(bbl/d), qw(bbl/d), pwf(psia)]
        """        
        qg = self._flow_gas_()
        qo = qg/self.go_ratio
        qw = (qg*self.water_cut*(1/self.go_ratio))/(1-self.water_cut)
        ql = qw + qo
        return [ql, qg/1000, qo, qw, self.delta_p]
    
if __name__ == "__main__":
    
    well = Darcy(1309, 600, 0.673, 11, 9.95, 76, 0.3542, 909, 0.88, 48.6)

    #print(well.flow_gas())
    
    # ql, qg, qo, qw, pw = well.inflow()
    # print(ql, qg, qo, qw, pw)
    
    # import matplotlib.pyplot as plt
    
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(ql, pw)
    # axs[0, 1].plot(qg, pw)
    # axs[1, 0].plot(qo, pw)
    # axs[1, 1].plot(qw, pw)
    # #plt.plot(qo, pw)
    # plt.show()

