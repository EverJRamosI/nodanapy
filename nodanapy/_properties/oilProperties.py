import petpropy as pp

class OilProperties:
    
    def __init__(self, pressure, temperature, specific_gravity, api, solution=None, bubble_pressure: int|float = 0):
        self.pressure = pressure
        self.temperature = temperature
        self.specific_gravity = specific_gravity
        self.api = api
        self.solution = solution
        self.bubble_pressure = bubble_pressure
        
    def bubble_pressure_oil(self, yN2=0, yCO2=0, yH2S=0):
        self.yN2 = yN2
        self.yCO2 = yCO2
        self.yH2S = yH2S        
        return pp.Pb.standing(self.solution, self.specific_gravity, self.temperature, self.api, y_N2=self.yN2, y_CO2=self.yCO2, y_H2S=self.yH2S)
    
    def solution_oil(self):
        return pp.Rs.standing(self.pressure, self.temperature, self.specific_gravity, self.api, Pb=self.bubble_pressure)
    
    def viscosity_oil(self):
        return pp.mu_o.mu_oil(self.pressure, self.temperature, self.api, self.solution_oil(), Pb=self.bubble_pressure)
    
    def factor_volumetric_oil(self, compressibility: float=0):
        return pp.Bo.glaso(self.temperature, self.solution_oil(), self.api, self.specific_gravity, P=self.pressure, Pb=self.bubble_pressure, co=compressibility)
    
    def density_oil(self):
        return pp.rho_o.standing(self.temperature, self.solution_oil(), self.specific_gravity, self.api)
    
    def tension_oil(self):
        return pp.sigma_o.baker_swedloff(self.pressure, self.temperature, self.api)
    
    
if __name__ == "__main__":
    
    mu = []
    rs = []
    p = list(range(0, 9500, 500))
    for i in p:
        oil = OilProperties(i, 544, 0.673, 53.7,)
        mui = oil.viscosity_oil()
        rsi = oil.solution_oil()
        mu.append(mui)
        rs.append(rsi)
    
    print(mu, rs, p)
    
    import matplotlib.pyplot as plt
    
    plt.plot(p, rs)
    plt.show()
    
    