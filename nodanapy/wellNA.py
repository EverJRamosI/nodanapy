import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d

from ipr.darcy import Darcy
from vlp.gray import Gray

class WellNA:
    def __init__(self, wellhead_pressure: int|float, wellhead_temperature: int|float, reservoir_pressure: int|float, reservoir_temperature: int|float, *,
                specific_gravity: float=0.65, permeability: int|float=10, skin: int|float=0, height_formation: int|float=10, 
                well_radius: int|float=0.35, reservoir_radius: int|float=1000, api: int|float=40, 
                bubble_pressure: int|float=0, salinity: int|float=1000, water_cut: float=0.1, gor: int|float=50, 
                internal_diameter: int|float=2.5, rugosity: float=0.0001, deep_well: int|float = 5000, amount: int=25):
        
        self.wellhead_pressure = wellhead_pressure
        self.wellhead_temperature = wellhead_temperature
        self.reservoir_pressure = reservoir_pressure
        self.reservoir_temperature = reservoir_temperature
        self.specific_gravity = specific_gravity
        self.permeability = permeability
        self.skin = skin
        self.height_formation = height_formation
        self.well_radius = well_radius
        self.reservoir_radius = reservoir_radius
        self.api = api
        self.bubble_pressure = bubble_pressure
        self.salinity = salinity
        self.water_cut = water_cut
        self.gor = gor
        self.internal_diameter = internal_diameter
        self.rugosity = rugosity
        self.deep_well = deep_well
        self.amount = amount
        
    def ipr(self):
        ipr = Darcy(self.reservoir_pressure, self.reservoir_temperature, self.specific_gravity, 
                    self.permeability, self.skin, self.height_formation, self.well_radius, 
                    self.reservoir_radius, self.water_cut, self.gor, self.amount)
        ql, qg, qo, qw, pwf = ipr.darcy()
        return {"Ql(bpd)": ql, "Qg(Mscfd)": qg, "Qo(bpd)": qo, "Qw(bpd)": qw, "Pwf(psia)": pwf}
    
    def vlp(self):
        aof = self.ipr()["Qg(Mscfd)"][-1]
        vlp = Gray(self.wellhead_pressure, self.wellhead_temperature, self.specific_gravity, self.api, 
                    self.bubble_pressure, self.salinity, self.water_cut, self.gor, self.internal_diameter, self.rugosity, self.deep_well, qg_max=aof, amount=self.amount)
        ql, qg, qo, qw, pwf = vlp.gray()
        return {"Ql(bpd)": ql, "Qg(Mscfd)": qg, "Qo(bpd)": qo, "Qw(bpd)": qw, "Pwf(psia)": pwf}

    def optimal_flow(self):
        
        ipr_values = self.ipr()
        ql_ipr = ipr_values["Ql(bpd)"]
        qg_ipr = ipr_values["Qg(Mscfd)"]
        qo_ipr = ipr_values["Qo(bpd)"]
        qw_ipr = ipr_values["Qw(bpd)"]
        pwf_ipr = ipr_values["Pwf(psia)"]
        
        vlp_values = self.vlp()
        ql_vlp = vlp_values["Ql(bpd)"]
        qg_vlp = vlp_values["Qg(Mscfd)"]
        qo_vlp = vlp_values["Qo(bpd)"]
        qw_vlp = vlp_values["Qw(bpd)"]
        pwf_vlp = vlp_values["Pwf(psia)"]
        
        def find_intersection(x_ipr, y_ipr, x_vlp, y_vlp):
            inter_ipr = interp1d(x_ipr, y_ipr, kind='cubic', fill_value='extrapolate')
            inter_vlp = interp1d(x_vlp, y_vlp, kind='cubic', fill_value='extrapolate')
            
            def func_to_solve(x):
                return inter_ipr(x) - inter_vlp(x)
            
            result = root_scalar(func_to_solve, bracket=[min(min(x_ipr), min(x_vlp)), max(max(x_ipr), max(x_vlp))])
            
            if result.converged:
                x_opt = result.root
                y_opt = inter_ipr(x_opt)
                return x_opt, y_opt
            else:
                print("No intersection found.")
                return None

        ql_opt, pwf_ql_opt = find_intersection(ql_ipr, pwf_ipr, ql_vlp, pwf_vlp)
        qg_opt, pwf_qg_opt = find_intersection(qg_ipr, pwf_ipr, qg_vlp, pwf_vlp)
        qo_opt, pwf_qo_opt = find_intersection(qo_ipr, pwf_ipr, qo_vlp, pwf_vlp)
        qw_opt, pwf_qw_opt = find_intersection(qw_ipr, pwf_ipr, qw_vlp, pwf_vlp)
        
        if None not in [ql_opt, qg_opt, qo_opt, qw_opt]:
            q_opt = [
                {"QlOpt(bpd)": np.array(ql_opt), "PwfQlOpt(psia)": pwf_ql_opt},
                {"QgOpt(Mscfd)": np.array(qg_opt), "PwfQgOpt(psia)": pwf_qg_opt},
                {"QoOpt(bpd)": np.array(qo_opt), "PwfQoOpt(psia)": pwf_qo_opt},
                {"QwOpt(bpd)": np.array(qw_opt), "PwfQwOpt(psia)": pwf_qw_opt}
            ]
            return q_opt
            
        else:
            print("No valid result was found for all variables.")
            return None
            
if __name__ == "__main__":
    
    well1 = WellNA(149.7, (84+460), 1309, (140+460), amount=5)

    print("IPR")
    print(well1.ipr())
    print("VLP")
    print(well1.vlp())
    print("Qopt")
    print(well1.optimal_flow())

# ipr = well1._ipr_()
# vlp = well1._vlp_()

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()

    # ax.plot(well1.ipr()["Qg(Mscfd)"], well1.ipr()["Pwf(psia)"])
    # ax.plot(well1.vlp()["Qg(Mscfd)"], well1.vlp()["Pwf(psia)"])

    # plt.show()