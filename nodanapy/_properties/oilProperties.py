import petpropy as pp

class OilProperties:
    
    def __init__(self, pressure, temperature, specific_gravity, api, bubble_pressure: int|float = 0):
        self.pressure = pressure
        self.temperature = temperature
        self.specific_gravity = specific_gravity
        self.api = api
        self.bubble_pressure = bubble_pressure
        
    def solution_oil(self):
        return pp.Rs.almarhoun(self.pressure, self.temperature, self.specific_gravity, self.api, Pb=self.bubble_pressure)
    
    def viscosity_oil(self):
        return pp.mu_o.mu_oil(self.pressure, self.temperature, self.api, self.solution_oil(), Pb=self.bubble_pressure)
    
    def factor_volumetric_oil(self, compressibility: float=0):
        return pp.Bo.glaso(self.temperature, self.solution_oil(), self.api, self.specific_gravity, P=self.pressure, Pb=self.bubble_pressure, co=compressibility)
    
    def density_oil(self):
        return pp.rho_o.standing(self.temperature, self.solution_oil(), self.specific_gravity, self.api)
    
    def tension_oil(self):
        return pp.sigma_o.baker_swedloff(self.pressure, self.temperature, self.api)
    
if __name__ == "__main__":
    oil = OilProperties(6500, 544, 0.673, 53.7, 149.7)
    print(oil.tension_oil())