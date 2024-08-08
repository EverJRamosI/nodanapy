import petpropy as pp

class GasProperties:
    
    def __init__(self, Pressure, Temperature, SpecificGravity):
        self.Pressure = Pressure
        self.Temperature = Temperature
        self.SpecificGravity = SpecificGravity
        
    def properties_critical_gas(self, yCO2: float=0, yH2S: float=0, yN2: float=0):
        return pp.pc_g.brown_katz_grv(self.SpecificGravity, yCO2=yCO2, yH2S=yH2S, yN2=yN2)
    
    def compressibility_gas(self):
        ppc, tpc = self.properties_critical_gas()
        return pp.z_g.dranchuk_purvis_robinson(self.Pressure, self.Temperature, ppc, tpc)
    
    def weight_molecular_gas(self, m_C7: float=0):
        if m_C7 != 0:
            return pp.m_g.gas_weight_molecular(m_C7=m_C7)
        else:
            return self.SpecificGravity * 28.97
    
    def viscosity_gas(self):
        return pp.mu_g.lee_gonzalez_eakin(self.Pressure, self.Temperature, self.weight_molecular_gas(), self.compressibility_gas())
    
    def density_gas(self):
        return pp.rho_g.rho_gas(self.Pressure, self.Temperature, self.SpecificGravity, self.compressibility_gas())
        
