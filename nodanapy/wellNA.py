import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from scipy._lib._bunch import _make_tuple_bunch

from ipr.darcy import Darcy
from ipr.lit import LITPD, LITRD
from ipr.vogel import VogelPD, VogelRD
from ipr.fetkovich import Fetkovich
from vlp.gray import Gray
from vlp.hagedornBrown import HagedornBrown
from vlp.beggsBrill import BeggsBrill

IPR = _make_tuple_bunch('IPR', ["Ql", "Qg", "Qo", "Qw", "Pwf"])
VLP = _make_tuple_bunch('VLP', ["Ql", "Qg", "Qo", "Qw", "Pwf"])
OPTIMAL_LIQUID = _make_tuple_bunch('LIQUID', ['Ql', 'Pwf'])
OPTIMAL_GAS = _make_tuple_bunch('GAS', ['Qg', 'Pwf'])
OPTIMAL_OIL = _make_tuple_bunch('OIL', ['Qo', 'Pwf'])
OPTIMAL_WATER = _make_tuple_bunch('WATER', ['Qw', 'Pwf'])
PRESSURE = _make_tuple_bunch('PWF', ['H', 'T', 'Vsl', 'Vsg', 'Vm', 'Hl', 'dP', 'P'])

class WellNAGasR:
    def __init__(self, wellhead_pressure: int|float, wellhead_temperature: int|float, reservoir_pressure: int|float, reservoir_temperature: int|float,
                specific_gravity: float=0.65, permeability: int|float=10, compressibility: float=1e-6, skin: int|float=0, height_formation: int|float=10, 
                well_radius: int|float=0.35, reservoir_radius: int|float=1000, api: int|float=40, 
                bubble_pressure: int|float=0, salinity: int|float=1000, water_cut: float=0.0, go_ratio: int|float=500, 
                internal_diameter: int|float=2.5, rugosity: float=0.0001, well_depth: int|float = 5000, amount: int=25,
                model_ipr: str='darcy', model_vlp: str='gray'):
        
        self.wellhead_pressure = wellhead_pressure
        self.wellhead_temperature = wellhead_temperature
        self.reservoir_pressure = reservoir_pressure
        self.reservoir_temperature = reservoir_temperature
        self.specific_gravity = specific_gravity
        self.permeability = permeability
        self.compressibility = compressibility
        self.skin = skin
        self.height_formation = height_formation
        self.well_radius = well_radius
        self.reservoir_radius = reservoir_radius
        self.api = api
        self.bubble_pressure = bubble_pressure
        self.salinity = salinity
        self.water_cut = water_cut
        self.go_ratio = go_ratio
        self.internal_diameter = internal_diameter
        self.rugosity = rugosity
        self.well_depth = well_depth
        self.amount = amount
        
        self.model_ipr = model_ipr
        self.model_vlp = model_vlp
        
        self.ipr = self._ipr_()
        self.vlp = self._vlp_()
        self.rate_liq, self.rate_gas, self.rate_oil, self.rate_water = self.optimal_flow()
        
    def _ipr_(self):
        
        if self.model_ipr == 'darcy':
            ipr = Darcy(self.reservoir_pressure, self.reservoir_temperature, self.specific_gravity, 
                        self.permeability, self.skin, self.height_formation, self.well_radius, 
                        self.reservoir_radius, self.water_cut, self.go_ratio, self.amount)
        elif self.model_ipr == 'LIT':
            ipr = LITRD(self.reservoir_pressure, self.reservoir_temperature, self.bubble_pressure, self.specific_gravity,
                        self.api, self.permeability, self.compressibility, self.skin, self.height_formation, self.well_radius, 
                        self.reservoir_radius, self.water_cut, self.go_ratio, self.amount)
        
        ql, qg, qo, qw, pwf = ipr.inflow()
        return IPR(Ql=ql, Qg=qg, Qo=qo, Qw=qw, Pwf=pwf)
        
    def _vlp_(self):
        qln = self.ipr.Ql[-1]
        
        if self.model_vlp == 'gray':
            vlp = Gray(pressure=self.wellhead_pressure, temperature=self.wellhead_temperature, specific_gravity=self.specific_gravity, api=self.api, 
                    bubble_pressure=self.bubble_pressure, salinity=self.salinity, water_cut=self.water_cut, go_ratio=self.go_ratio, internal_diameter=self.internal_diameter, 
                    rugosity=self.rugosity, well_depth=self.well_depth, temperature_node=self.reservoir_temperature, ql_i=0.001, ql_n=qln, amount=self.amount)
        
        ql, qg, qo, qw, pwf = vlp.outflow()
        return VLP(Ql=ql, Qg=qg, Qo=qo, Qw=qw, Pwf=pwf)
        
    def optimal_flow(self):
        
        ql_ipr = self.ipr.Ql
        qg_ipr = self.ipr.Qg
        qo_ipr = self.ipr.Qo
        qw_ipr = self.ipr.Qw
        pwf_ipr = self.ipr.Pwf
        
        ql_vlp = self.vlp.Ql 
        qg_vlp = self.vlp.Qg
        qo_vlp = self.vlp.Qo
        qw_vlp = self.vlp.Qw
        pwf_vlp = self.vlp.Pwf
        
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

        ql_opt, pwf_l_opt = find_intersection(ql_ipr, pwf_ipr, ql_vlp, pwf_vlp)
        qg_opt, pwf_g_opt = find_intersection(qg_ipr, pwf_ipr, qg_vlp, pwf_vlp)
        qo_opt, pwf_o_opt = find_intersection(qo_ipr, pwf_ipr, qo_vlp, pwf_vlp)
        qw_opt, pwf_w_opt = None, None
        if all(qw_ipr)!=0 or all(qw_vlp)!=0:
            qw_opt, pwf_w_opt = find_intersection(qw_ipr, pwf_ipr, qw_vlp, pwf_vlp)
        
        if None not in [ql_opt, pwf_l_opt] or None not in [qg_opt, pwf_g_opt] or None not in [qo_opt, pwf_o_opt] or None not in [qw_opt, pwf_w_opt]:
            if ql_opt!=0 and pwf_l_opt!=0:
                liquid = OPTIMAL_LIQUID(Ql=np.array(ql_opt), Pwf=pwf_l_opt)
            else:
                liquid = OPTIMAL_LIQUID(Ql=0, Pwf=0)
            if qg_opt!=0 and pwf_g_opt!=0:
                gas = OPTIMAL_GAS(Qg=np.array(qg_opt), Pwf=pwf_g_opt)
            else:
                gas = OPTIMAL_GAS(Qg=0, Pwf=0)
            if qo_opt!=0 and pwf_o_opt!=0:
                oil = OPTIMAL_OIL(Qo=np.array(ql_opt), Pwf=pwf_l_opt)
            else:
                oil = OPTIMAL_OIL(Qo=0, Pwf=0)
            if qw_opt!=0 and pwf_w_opt!=0 and qw_opt is not None and pwf_w_opt is not None:
                water = OPTIMAL_WATER(Qw=np.array(qw_opt), Pwf=pwf_w_opt)
            else:
                water = OPTIMAL_WATER(Qw=0, Pwf=0)
                
        return liquid, gas, oil, water
    
    def pressure_flow(self):
        qli = self.rate_liq.Ql
        
        optimal = Gray(pressure=self.wellhead_pressure, temperature=self.wellhead_temperature, specific_gravity=self.specific_gravity, api=self.api, 
                    bubble_pressure=self.bubble_pressure, salinity=self.salinity, water_cut=self.water_cut, go_ratio=self.go_ratio, internal_diameter=self.internal_diameter, 
                    rugosity=self.rugosity, well_depth=self.well_depth, temperature_node=self.reservoir_temperature, ql_i=qli, amount=self.amount)
        
        h = optimal.delta_depth
        t = optimal.delta_t
        vsl, vsg, vm, hl, dp, p = optimal.pressure_traverse()
        
        return PRESSURE(H=h, T=t, Vsl=vsl, Vsg=vsg, Vm=vm, Hl=hl, dP=dp, P=p)

class WellNAGasP:
    def __init__(self, wellhead_pressure: int|float, wellhead_temperature: int|float, reservoir_pressure: int|float, reservoir_temperature: int|float,
                qg_test: list=[3000, 4500, 5600, 6300], pwf_test: list=[3500, 3000, 2500, 2000], api: int|float=40, specific_gravity: float=0.65, bubble_pressure: int|float=0, salinity: int|float=1000, water_cut: float=0.0, go_ratio: int|float=500, 
                internal_diameter: int|float=2.5, rugosity: float=0.0001, well_depth: int|float = 5000, amount: int=25,
                model_ipr: str='LIT', model_vlp: str='gray'):
        
        self.wellhead_pressure = wellhead_pressure
        self.wellhead_temperature = wellhead_temperature
        self.reservoir_pressure = reservoir_pressure
        self.reservoir_temperature = reservoir_temperature
        self.qg_test = qg_test
        self.pwf_test = pwf_test
        self.specific_gravity = specific_gravity
        self.api = api
        self.bubble_pressure = bubble_pressure
        self.salinity = salinity
        self.water_cut = water_cut
        self.go_ratio = go_ratio
        self.internal_diameter = internal_diameter
        self.rugosity = rugosity
        self.well_depth = well_depth
        self.amount = amount
        
        self.model_ipr = model_ipr
        self.model_vlp = model_vlp
        
        self.ipr = self._ipr_()
        self.vlp = self._vlp_()
        self.rate_liq, self.rate_gas, self.rate_oil, self.rate_water = self.optimal_flow()
        
    def _ipr_(self):
        
        if self.model_ipr == 'LIT':
            ipr = LITPD(self.reservoir_pressure, self.reservoir_temperature, self.qg_test, self.pwf_test, self.specific_gravity,
                        self.water_cut, self.go_ratio, self.amount)
        
        ql, qg, qo, qw, pwf = ipr.inflow()
        return IPR(Ql=ql, Qg=qg, Qo=qo, Qw=qw, Pwf=pwf)
        
    def _vlp_(self):
        qln = self.ipr.Ql[-1]
        
        if self.model_vlp == 'gray':
            vlp = Gray(pressure=self.wellhead_pressure, temperature=self.wellhead_temperature, specific_gravity=self.specific_gravity, api=self.api, 
                    bubble_pressure=self.bubble_pressure, salinity=self.salinity, water_cut=self.water_cut, go_ratio=self.go_ratio, internal_diameter=self.internal_diameter, 
                    rugosity=self.rugosity, well_depth=self.well_depth, temperature_node=self.reservoir_temperature, ql_i=0.001, ql_n=qln, amount=self.amount)
        
        ql, qg, qo, qw, pwf = vlp.outflow()
        return VLP(Ql=ql, Qg=qg, Qo=qo, Qw=qw, Pwf=pwf)
        
    def optimal_flow(self):
        
        ql_ipr = self.ipr.Ql
        qg_ipr = self.ipr.Qg
        qo_ipr = self.ipr.Qo
        qw_ipr = self.ipr.Qw
        pwf_ipr = self.ipr.Pwf
        
        ql_vlp = self.vlp.Ql 
        qg_vlp = self.vlp.Qg
        qo_vlp = self.vlp.Qo
        qw_vlp = self.vlp.Qw
        pwf_vlp = self.vlp.Pwf
        
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

        ql_opt, pwf_l_opt = find_intersection(ql_ipr, pwf_ipr, ql_vlp, pwf_vlp)
        qg_opt, pwf_g_opt = find_intersection(qg_ipr, pwf_ipr, qg_vlp, pwf_vlp)
        qo_opt, pwf_o_opt = find_intersection(qo_ipr, pwf_ipr, qo_vlp, pwf_vlp)
        qw_opt, pwf_w_opt = None, None
        if all(qw_ipr)!=0 or all(qw_vlp)!=0:
            qw_opt, pwf_w_opt = find_intersection(qw_ipr, pwf_ipr, qw_vlp, pwf_vlp)
        
        if None not in [ql_opt, pwf_l_opt] or None not in [qg_opt, pwf_g_opt] or None not in [qo_opt, pwf_o_opt] or None not in [qw_opt, pwf_w_opt]:
            if ql_opt!=0 and pwf_l_opt!=0:
                liquid = OPTIMAL_LIQUID(Ql=np.array(ql_opt), Pwf=pwf_l_opt)
            else:
                liquid = OPTIMAL_LIQUID(Ql=0, Pwf=0)
            if qg_opt!=0 and pwf_g_opt!=0:
                gas = OPTIMAL_GAS(Qg=np.array(qg_opt), Pwf=pwf_g_opt)
            else:
                gas = OPTIMAL_GAS(Qg=0, Pwf=0)
            if qo_opt!=0 and pwf_o_opt!=0:
                oil = OPTIMAL_OIL(Qo=np.array(ql_opt), Pwf=pwf_l_opt)
            else:
                oil = OPTIMAL_OIL(Qo=0, Pwf=0)
            if qw_opt!=0 and pwf_w_opt!=0 and qw_opt is not None and pwf_w_opt is not None:
                water = OPTIMAL_WATER(Qw=np.array(qw_opt), Pwf=pwf_w_opt)
            else:
                water = OPTIMAL_WATER(Qw=0, Pwf=0)
                
        return liquid, gas, oil, water
    
    def pressure_flow(self):
        qli = self.rate_liq.Ql
        
        optimal = Gray(pressure=self.wellhead_pressure, temperature=self.wellhead_temperature, specific_gravity=self.specific_gravity, api=self.api, 
                    bubble_pressure=self.bubble_pressure, salinity=self.salinity, water_cut=self.water_cut, go_ratio=self.go_ratio, internal_diameter=self.internal_diameter, 
                    rugosity=self.rugosity, well_depth=self.well_depth, temperature_node=self.reservoir_temperature, ql_i=qli, amount=self.amount)
        
        h = optimal.delta_depth
        t = optimal.delta_t
        vsl, vsg, vm, hl, dp, p = optimal.pressure_traverse()
        
        return PRESSURE(H=h, T=t, Vsl=vsl, Vsg=vsg, Vm=vm, Hl=hl, dP=dp, P=p)            
class WellNAOilR:
    def __init__(self, wellhead_pressure: int|float, wellhead_temperature: int|float, reservoir_pressure: int|float, reservoir_temperature: int|float,
                bubble_pressure: int|float=0, specific_gravity: float=0.65, permeability: int|float=10, skin: int|float=0, compressibility: float=1e-6,
                height_formation: int|float=10, well_radius: int|float=0.35, reservoir_radius: int|float=1000, api: int|float=40, 
                salinity: int|float=1000, water_cut: float=0.0, go_ratio: int|float=500, 
                internal_diameter: int|float=2.5, rugosity: float=0.0001, well_depth: int|float = 5000, amount: int=25,
                model_ipr: str='vogel', model_vlp: str='hagedorn'):
        
        self.wellhead_pressure = wellhead_pressure
        self.wellhead_temperature = wellhead_temperature
        self.reservoir_pressure = reservoir_pressure
        self.reservoir_temperature = reservoir_temperature
        self.bubble_pressure = bubble_pressure
        self.specific_gravity = specific_gravity
        self.permeability = permeability
        self.compressibility = compressibility
        self.skin = skin
        self.height_formation = height_formation
        self.well_radius = well_radius
        self.reservoir_radius = reservoir_radius
        self.api = api
        self.salinity = salinity
        self.water_cut = water_cut
        self.go_ratio = go_ratio
        self.internal_diameter = internal_diameter
        self.rugosity = rugosity
        self.well_depth = well_depth
        self.amount = amount
        
        self.model_ipr = model_ipr
        self.model_vlp = model_vlp
        
        self.ipr = self._ipr_()
        self.vlp = self._vlp_()
        self.rate_liq, self.rate_gas, self.rate_oil, self.rate_water = self.optimal_flow()
    
    def _ipr_(self):
        
        if self.model_ipr == 'vogel':
            ipr = VogelRD(self.reservoir_pressure, self.reservoir_temperature, self.bubble_pressure, self.specific_gravity, self.api,
                    self.permeability, self.compressibility, self.skin, self.height_formation, self.well_radius, 
                    self.reservoir_radius, self.water_cut, self.go_ratio, self.amount)
        
        ql, qg, qo, qw, pwf = ipr.inflow()
        return IPR(Ql=ql, Qg=qg, Qo=qo, Qw=qw, Pwf=pwf)
        
    def _vlp_(self):
        qli = 0.001
        qln = self.ipr.Ql[-1]
        
        if self.model_vlp == 'hagedorn':
            vlp = HagedornBrown(self.wellhead_pressure, self.wellhead_temperature, self.specific_gravity, self.api, 
                    self.bubble_pressure, self.salinity, self.water_cut, self.go_ratio, self.internal_diameter, 
                    self.rugosity, self.well_depth, self.reservoir_temperature, qli, qln, amount=self.amount)
        elif self.model_vlp == 'beggs':
            vlp = BeggsBrill(self.wellhead_pressure, self.wellhead_temperature, self.specific_gravity, self.api, 
                    self.bubble_pressure, self.salinity, self.water_cut, self.go_ratio, self.internal_diameter, 
                    self.rugosity, self.well_depth, self.reservoir_temperature, qo_i=qli, qo_n=qln, amount=self.amount)
        
        ql, qg, qo, qw, pwf = vlp.outflow()
        return VLP(Ql=ql, Qg=qg, Qo=qo, Qw=qw, Pwf=pwf)            
            
    def optimal_flow(self):
        
        ql_ipr = self.ipr.Ql
        qg_ipr = self.ipr.Qg
        qo_ipr = self.ipr.Qo
        qw_ipr = self.ipr.Qw
        pwf_ipr = self.ipr.Pwf
        
        ql_vlp = self.vlp.Ql 
        qg_vlp = self.vlp.Qg
        qo_vlp = self.vlp.Qo
        qw_vlp = self.vlp.Qw
        pwf_vlp = self.vlp.Pwf
        
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

        ql_opt, pwf_l_opt = find_intersection(ql_ipr, pwf_ipr, ql_vlp, pwf_vlp)
        qg_opt, pwf_g_opt = find_intersection(qg_ipr, pwf_ipr, qg_vlp, pwf_vlp)
        qo_opt, pwf_o_opt = find_intersection(qo_ipr, pwf_ipr, qo_vlp, pwf_vlp)
        qw_opt, pwf_w_opt = None, None
        if all(qw_ipr)!=0 or all(qw_vlp)!=0:
            qw_opt, pwf_w_opt = find_intersection(qw_ipr, pwf_ipr, qw_vlp, pwf_vlp)
        
        if None not in [ql_opt, pwf_l_opt] or None not in [qg_opt, pwf_g_opt] or None not in [qo_opt, pwf_o_opt] or None not in [qw_opt, pwf_w_opt]:
            if ql_opt!=0 and pwf_l_opt!=0:
                liquid = OPTIMAL_LIQUID(Ql=np.array(ql_opt), Pwf=pwf_l_opt)
            else:
                liquid = OPTIMAL_LIQUID(Ql=0, Pwf=0)
            if qg_opt!=0 and pwf_g_opt!=0:
                gas = OPTIMAL_GAS(Qg=np.array(qg_opt), Pwf=pwf_g_opt)
            else:
                gas = OPTIMAL_GAS(Qg=0, Pwf=0)
            if qo_opt!=0 and pwf_o_opt!=0:
                oil = OPTIMAL_OIL(Qo=np.array(ql_opt), Pwf=pwf_l_opt)
            else:
                oil = OPTIMAL_OIL(Qo=0, Pwf=0)
            if qw_opt!=0 and pwf_w_opt!=0 and qw_opt is not None and pwf_w_opt is not None:
                water = OPTIMAL_WATER(Qw=np.array(qw_opt), Pwf=pwf_w_opt)
            else:
                water = OPTIMAL_WATER(Qw=0, Pwf=0)
                
        return liquid, gas, oil, water
    
    def pressure_flow(self):
        qli = self.rate_liq.Ql
        
        if self.model_vlp == 'hagedorn':
            optimal = HagedornBrown(self.wellhead_pressure, self.wellhead_temperature, self.specific_gravity, self.api, 
                    self.bubble_pressure, self.salinity, self.water_cut, self.go_ratio, self.internal_diameter, 
                    self.rugosity, self.well_depth, self.reservoir_temperature, qli, amount=self.amount)
        elif self.model_vlp == 'beggs':
            optimal = BeggsBrill(self.wellhead_pressure, self.wellhead_temperature, self.specific_gravity, self.api, 
                    self.bubble_pressure, self.salinity, self.water_cut, self.go_ratio, self.internal_diameter, 
                    self.rugosity, self.well_depth, self.reservoir_temperature, qo_i=qli,  amount=self.amount)
        
        h = optimal.delta_depth
        t = optimal.delta_t
        vsl, vsg, vm, hl, dp, p = optimal.pressure_traverse()
        
        return PRESSURE(H=h, T=t, Vsl=vsl, Vsg=vsg, Vm=vm, Hl=hl, dP=dp, P=p)

class WellNAOilP:
    def __init__(self, wellhead_pressure: int|float, wellhead_temperature: int|float, reservoir_pressure: int|float, reservoir_temperature: int|float,
                bubble_pressure: int|float=0, qo_test: list|int|float=None, pwf_test: list|int|float=None,
                specific_gravity: float=0.65, api: int|float=40, salinity: int|float=1000, water_cut: float=0.0, go_ratio: int|float=500, 
                internal_diameter: int|float=2.5, rugosity: float=0.0001, well_depth: int|float = 5000, amount: int=25,
                model_ipr: str='vogel', model_vlp: str='hagedorn'):
        
        self.wellhead_pressure = wellhead_pressure
        self.wellhead_temperature = wellhead_temperature
        self.reservoir_pressure = reservoir_pressure
        self.reservoir_temperature = reservoir_temperature
        self.bubble_pressure = bubble_pressure
        self.qo_test = qo_test
        self.pwf_test = pwf_test
        self.specific_gravity = specific_gravity
        self.api = api
        self.salinity = salinity
        self.water_cut = water_cut
        self.go_ratio = go_ratio
        self.internal_diameter = internal_diameter
        self.rugosity = rugosity
        self.well_depth = well_depth
        self.amount = amount
        
        self.model_ipr = model_ipr
        self.model_vlp = model_vlp
        
        self.ipr = self._ipr_()
        self.vlp = self._vlp_()
        self.rate_liq, self.rate_gas, self.rate_oil, self.rate_water = self.optimal_flow()
    
    def _ipr_(self):
        
        if self.model_ipr == 'vogel':
            ipr = VogelPD(self.reservoir_pressure, self.bubble_pressure, self.qo_test, self.pwf_test, self.water_cut, self.go_ratio, self.amount)
        elif self.model_ipr == 'fetkovich':
            ipr = Fetkovich(self.reservoir_pressure, self.qo_test, self.pwf_test, self.water_cut, self.go_ratio, self.amount)
        
        ql, qg, qo, qw, pwf = ipr.inflow()
        return IPR(Ql=ql, Qg=qg, Qo=qo, Qw=qw, Pwf=pwf)
        
    def _vlp_(self):
        qli = 0.001
        qln = self.ipr.Ql[-1]
        
        if self.model_vlp == 'hagedorn':
            vlp = HagedornBrown(self.wellhead_pressure, self.wellhead_temperature, self.specific_gravity, self.api, 
                    self.bubble_pressure, self.salinity, self.water_cut, self.go_ratio, self.internal_diameter, 
                    self.rugosity, self.well_depth, self.reservoir_temperature, qli, qln, amount=self.amount)
        elif self.model_vlp == 'beggs':
            vlp = BeggsBrill(self.wellhead_pressure, self.wellhead_temperature, self.specific_gravity, self.api, 
                    self.bubble_pressure, self.salinity, self.water_cut, self.go_ratio, self.internal_diameter, 
                    self.rugosity, self.well_depth, self.reservoir_temperature, qo_i=qli, qo_n=qln, amount=self.amount)
        
        ql, qg, qo, qw, pwf = vlp.outflow()
        return VLP(Ql=ql, Qg=qg, Qo=qo, Qw=qw, Pwf=pwf) 
            
    def optimal_flow(self):
        
        ql_ipr = self.ipr.Ql
        qg_ipr = self.ipr.Qg
        qo_ipr = self.ipr.Qo
        qw_ipr = self.ipr.Qw
        pwf_ipr = self.ipr.Pwf
        
        ql_vlp = self.vlp.Ql 
        qg_vlp = self.vlp.Qg
        qo_vlp = self.vlp.Qo
        qw_vlp = self.vlp.Qw
        pwf_vlp = self.vlp.Pwf
        
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

        ql_opt, pwf_l_opt = find_intersection(ql_ipr, pwf_ipr, ql_vlp, pwf_vlp)
        qg_opt, pwf_g_opt = find_intersection(qg_ipr, pwf_ipr, qg_vlp, pwf_vlp)
        qo_opt, pwf_o_opt = find_intersection(qo_ipr, pwf_ipr, qo_vlp, pwf_vlp)
        qw_opt, pwf_w_opt = None, None
        if all(qw_ipr)!=0 or all(qw_vlp)!=0:
            qw_opt, pwf_w_opt = find_intersection(qw_ipr, pwf_ipr, qw_vlp, pwf_vlp)
        
        if None not in [ql_opt, pwf_l_opt] or None not in [qg_opt, pwf_g_opt] or None not in [qo_opt, pwf_o_opt] or None not in [qw_opt, pwf_w_opt]:
            if ql_opt!=0 and pwf_l_opt!=0:
                liquid = OPTIMAL_LIQUID(Ql=np.array(ql_opt), Pwf=pwf_l_opt)
            else:
                liquid = OPTIMAL_LIQUID(Ql=0, Pwf=0)
            if qg_opt!=0 and pwf_g_opt!=0:
                gas = OPTIMAL_GAS(Qg=np.array(qg_opt), Pwf=pwf_g_opt)
            else:
                gas = OPTIMAL_GAS(Qg=0, Pwf=0)
            if qo_opt!=0 and pwf_o_opt!=0:
                oil = OPTIMAL_OIL(Qo=np.array(ql_opt), Pwf=pwf_l_opt)
            else:
                oil = OPTIMAL_OIL(Qo=0, Pwf=0)
            if qw_opt!=0 and pwf_w_opt!=0 and qw_opt is not None and pwf_w_opt is not None:
                water = OPTIMAL_WATER(Qw=np.array(qw_opt), Pwf=pwf_w_opt)
            else:
                water = OPTIMAL_WATER(Qw=0, Pwf=0)
                
        return liquid, gas, oil, water
    
    def pressure_flow(self):
        qli = self.rate_liq.Ql
        
        if self.model_vlp == 'hagedorn':
            optimal = HagedornBrown(self.wellhead_pressure, self.wellhead_temperature, self.specific_gravity, self.api, 
                    self.bubble_pressure, self.salinity, self.water_cut, self.go_ratio, self.internal_diameter, 
                    self.rugosity, self.well_depth, self.reservoir_temperature, ql_i=qli, amount=self.amount)
        elif self.model_vlp == 'beggs':
            optimal = BeggsBrill(self.wellhead_pressure, self.wellhead_temperature, self.specific_gravity, self.api, 
                    self.bubble_pressure, self.salinity, self.water_cut, self.go_ratio, self.internal_diameter, 
                    self.rugosity, self.well_depth, self.reservoir_temperature, ql_i=qli, amount=self.amount)
        
        h = optimal.delta_depth
        t = optimal.delta_t
        vsl, vsg, vm, hl, dp, p = optimal.pressure_traverse()
        
        return PRESSURE(H=h, T=t, Vsl=vsl, Vsg=vsg, Vm=vm, Hl=hl, dP=dp, P=p)

if __name__ == "__main__":
    #import time
    #time_start = time.time()
    #well = WellNAOil(400, (84+460), 4500, (140+460), internal_diameter=4.5, well_depth=6000, bubble_pressure=1500)# qo_test=[300, 400, 500, 630], pwf_test=[3500, 3000, 2500, 2000], model_ipr='fetkovich', model_vlp='beggs')
    well1 = WellNAOilR(149.7, (84+460), 4500, (140+460), internal_diameter=1.5, well_depth=6000, bubble_pressure=1000,)# qo_test=[300, 400, 500, 630], pwf_test=[3500, 3000, 2500, 2000], model_ipr='fetkovich', model_vlp='beggs')
    
    #well = WellNAGas(400, (84+460), 4500, (140+460), go_ratio=50, internal_diameter=1.5)
    #well1 = WellNAGas(140, (84+460), 1300, (140+460), go_ratio=5000, internal_diameter=2.5, water_cut=0.85)
    
    #print("IPR")
    #print(well.ipr)
    #print("VLP")
    #print(well.vlp)
    #ipr_well = well.ipr
    #vlp_well = well.vlp
    #print('IPR', ipr_well, 'VLP', vlp_well)

    # ipr_well1 = well1.ipr
    # vlp_well1 = well1.vlp
    # print('IPR', ipr_well1, 'VLP', vlp_well1)
    # print('Optimal', well1.optimal_flow())
    # print('Pressure', well1.pressure_flow())
    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(2, 2)

    # # #ax.plot(ipr.Qo, ipr.Pwf)
    # # #ax.plot(vlp.Qo, vlp.Pwf)
    
    # #ax[0, 0].plot(ipr_well.Ql, ipr_well.Pwf)
    # #ax[0, 0].plot(vlp_well.Ql, vlp_well.Pwf)
    # ax[0, 0].plot(ipr_well1.Ql, ipr_well1.Pwf)
    # ax[0, 0].plot(vlp_well1.Ql, vlp_well1.Pwf)
    
    # #ax[0, 1].plot(ipr_well.Qg, ipr_well.Pwf)
    # #ax[0, 1].plot(vlp_well.Qg, vlp_well.Pwf)
    # ax[0, 1].plot(ipr_well1.Qg, ipr_well1.Pwf)
    # ax[0, 1].plot(vlp_well1.Qg, vlp_well1.Pwf)
    
    # #ax[1, 0].plot(ipr_well.Qo, ipr_well.Pwf)
    # #ax[1, 0].plot(vlp_well.Qo, vlp_well.Pwf)
    # ax[1, 0].plot(ipr_well1.Qo, ipr_well1.Pwf)
    # ax[1, 0].plot(vlp_well1.Qo, vlp_well1.Pwf)
    
    # ax[1, 1].plot(ipr_well1.Qw, ipr_well1.Pwf)
    # ax[1, 1].plot(vlp_well1.Qw, vlp_well1.Pwf)

    # time_end = time.time()
    # print('time', (time_end-time_start))
    # plt.show()