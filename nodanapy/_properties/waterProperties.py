import petpropy as pp

class WaterProperties:
    
    def __init__(self, pressure, temperature, salinity):
        self.temperature = temperature
        self.pressure = pressure
        self.salinity = salinity
    
    def viscosity_water(self):
        return pp.mu_w.mccain(self.pressure, self.temperature, S=self.salinity)
    
    def factor_volumetric_water(self):
        return pp.Bw.mccain(self.pressure, self.temperature)
    
    def density_water(self):
        return pp.rho_w.mccain(self.factor_volumetric_water(), S=self.salinity)
    
    def tension_water(self):
        return pp.sigma_w.jennings_newman(self.pressure, self.temperature)
    
    
    
#water1 = WaterProperties(149.7, (85+460), 8415)

#print(water1.density_water())