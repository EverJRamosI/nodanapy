from typing import List
import numpy as np

class Fetkovich:
    """
    ### Summary:
    This class is to determine IPR with the Fetkovich method for oil well with production data.
    
    ### Methods:
    - __linear_regression_
    - __n_
    - inflow: This method is to calculate the IPR.
    """    
    def __init__(self, pressure: int|float, qo_test: List[int|float]=[300, 450, 560, 630], pwf_test: List[int|float]=[3500, 3000, 2500, 2000], 
                water_cut: float=0.0, go_ratio: int|float=50, amount: int=25):
        """
        Args:
            pressure (int | float): Well Pressure [psia]
            qo_test (List[int | float]): Oil Rate Test [bbl/d]. Defaults to [300, 450, 560, 630].
            pwf_test (List[int | float]): Flowing Pressure Test [psia]. Defaults to [3500, 3000, 2500, 2000].
            water_cut (float): Water Cut. Defaults to 0.0.
            go_ratio (int | float): Gas-Oil Ratio [scf/stb]. Defaults to 50.
            amount (int, optional): Number of Points. Defaults to 25.
        ### Private Args:
            delta_p (array): Pressures Assumed from 14.7 to well pressure [psia]
            _a (float): Intersection
            _b (float): Slope
            n (float): Constant n
            q_max (float): Max Rate [bbl/d]
            c (float): Constant c
        """
        
        self.pressure = pressure
        self.qo_test = np.array(qo_test)
        self.pwf_test = np.array(pwf_test)
        self.water_cut = water_cut
        self.go_ratio = go_ratio
        self.amount = amount
        
        self.delta_p = np.linspace(self.pressure, 14.7, self.amount)
        
        self._a, self._b = self._linear_regression_()
        
        self.n = np.abs(self._n_())
        self.q_max = (self.pressure**2-self._b)/self._a
        self.c = self.q_max/((self.pressure**2)**self.n)
        
    def _linear_regression_(self):
        delta_p = self.pressure**2-self.pwf_test**2
        q2 = self.qo_test**2
        q_dp = self.qo_test*delta_p
        
        a = (len(self.pwf_test)*np.sum(q_dp) - np.sum(self.qo_test)*np.sum(delta_p))/(len(self.pwf_test)*np.sum(q2)-(np.sum(self.qo_test))**2)
        b = (np.sum(delta_p)*np.sum(q2) - np.sum(self.qo_test)*np.sum(q_dp))/(len(self.pwf_test)*np.sum(q2)-(np.sum(self.qo_test))**2)
        
        return a, b

    def _n_(self):
        delta_p = self.pressure**2-self.pwf_test**2
        return (np.log10(self.qo_test[-1])-np.log10(self.qo_test[0]))/(np.log10(delta_p[-1])-np.log10(delta_p[0]))
    
    def inflow(self):
        """
        Returns:
            List: It will return a list with the rates and flowing bottom hole pressure\n
            [ql (bbl/d), qg(Mscf/d), qo(bbl/d), qw(bbl/d), pwf(psia)]
        """        
        qo = self.c*((self.pressure**2 - self.delta_p**2)**self.n)
        qg = qo*self.go_ratio
        qw = qo*self.water_cut/(1-self.water_cut)
        ql = qo + qw
        return [ql, qg/1000, qo, qw, self.delta_p]
    
if __name__ == "__main__":
    well = Fetkovich(3600, [263, 383, 467, 640], [3170, 2890, 2440, 2150])
    
    #print(well.qo_test, well.pwf_test)
    # print(well._a, well._b, well.q_max, well.n, well.c)
    # #print(well.inflow())
    # print('linear 1', well._linear_regression_())
    # print('linear 2', well.linear())
    
    
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