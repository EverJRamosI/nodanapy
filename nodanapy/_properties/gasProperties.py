import petpropy as pp

class GasProperties:
    
    def __init__(self, pressure, temperature, specific_gravity):
        self.pressure = pressure
        self.temperature = temperature
        self.specific_gravity = specific_gravity
        
    def properties_critical_gas(self, yCO2: float=0, yH2S: float=0, yN2: float=0):
        return pp.pc_g.brown_katz_grv(self.specific_gravity, yCO2=yCO2, yH2S=yH2S, yN2=yN2)
    
    def factor_compressibility_gas(self):
        ppc, tpc = self.properties_critical_gas()
        return pp.z_g.dranchuk_purvis_robinson(self.pressure, self.temperature, ppc, tpc)
    
    def weight_molecular_gas(self, m_C7: float=0):
        if m_C7 != 0:
            return pp.m_g.gas_weight_molecular(m_C7=m_C7)
        else:
            return self.specific_gravity * 28.97
    
    def viscosity_gas(self):
        return pp.mu_g.lee_gonzalez_eakin(self.pressure, self.temperature, self.weight_molecular_gas(), self.factor_compressibility_gas())
    
    def density_gas(self):
        return pp.rho_g.rho_gas(self.pressure, self.temperature, self.specific_gravity, self.factor_compressibility_gas())
        

