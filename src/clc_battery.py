from vessim import Storage


class CLCBattery(Storage):
    """CLC Battery model for lithium-ion batteries. Default is the LGM50 21700 parameterization. 
    
    Args:
        energy_capacity: Single cell battery capacity in Wh. Default is 18.2Wh.
        initial_soc: Initial battery state-of-charge. Has to be between 0 and 1. Defaults to 0.
        nom_voltage: Single cell nominal voltage in V. Defaults to 3.63V.
        alpha_d: Maximum discharging C-rate. Defaults to 3.0C.
        alpha_c: Maximum charging C-rate. Defaults to 0.7C.
        eta_d: Average fraction of power that has to be discharged from to obtain said power.
            Is equivalent to the discharging inefficiency. Defaults to 1.043.
        eta_c: Average fraction of power that is stored in battery when charged at said power.
            Is equivalent to the charging inefficiency. Defualts to 0.979.
        u_1: Linear factor for the lower state-of-charge limit depending on the applied discharge rate.
            Defaults to 0.069.
        v_1: Offset for the lower state-of-charge limit depending on the applied discharge rate.
            Defaults to 0.0.
        u_2: Linear factor for the upper state-of-charge limit depending on the applied charge rate.
            Defaults to -0.081.
        v_2: Offset for the upper state-of-charge limit depending on the applied charge rate.
            Defaults to 1.
    """

    def __init__(
        self,
        energy_capacity: float = 18.2,
        initial_soc: float = 0,
        nom_voltage: float = 3.63,
        alpha_d: float = 3.0,
        alpha_c: float = 0.7,
        eta_d: float = 1.043,
        eta_c: float = 0.979,
        u_1: float = 0.069,
        v_1: float = 0.0,
        u_2: float = -0.081,
        v_2: float = 1.0,    
    ) -> None:
        self.energy_capacity = energy_capacity # Wh
        assert 0 <= initial_soc <= 1, "Invalid intitial state-of-charge. Has to be between 0 and 1."
        self.charge_level = energy_capacity * initial_soc # Wh
        self.nom_voltage = nom_voltage # V
        self.capacity = self.energy_capacity / self.nom_voltage # A
        self.alpha_d = - alpha_d * self.capacity # C-Rate to A
        self.alpha_c = alpha_c * self.capacity # C-Rate to A
        self.eta_d = eta_d
        self.eta_c = eta_c
        self.u_1 = u_1 * self.nom_voltage # C-Rate at SoC -> A at Wh
        self.v_1 = v_1 * self.energy_capacity # C-Rate at SoC -> A at Wh
        self.u_2 = u_2 * self.nom_voltage # C-Rate at SoC -> A at Wh
        self.v_2 = v_2 * self.energy_capacity # C-Rate at SoC -> A at Wh
    
    def soc(self) -> float:
        return self.charge_level / self.energy_capacity

    def update(self, power: float, duration: int) -> float:
        applied_power = power
        current = applied_power/ self.nom_voltage
        if current < self.alpha_d:
            print("Discharging current exceeds maximum discharge rate.")
            current = self.alpha_d
            applied_power = current * self.nom_voltage
        elif current > self.alpha_c:
            print("Charging current exceeds maximum charging rate.")
            current = self.alpha_c
            applied_power = current * self.nom_voltage
        
        if applied_power > 0:
            return self.charge(applied_power, current, duration)
        elif applied_power < 0:
            return self.discharge(applied_power, current, duration)
        else:
            return 0
    
    def charge(self, power: float, current: float, duration: int) -> float:
        energy_limit = self.u_2 * current + self.v_2
        maximum_duration = (energy_limit - self.charge_level) / (self.eta_c * power)
        if maximum_duration < duration:
            applied_duration = maximum_duration
        else:
            applied_duration = duration
        self.charge_level += self.eta_c * power * applied_duration
        return power * applied_duration
        
    def discharge(self, power: float, current: float, duration: int) -> float:
        energy_limit = self.u_1 * (-current) + self.v_1
        maximum_duration = (energy_limit - self.charge_level) / (self.eta_d * power)
        if maximum_duration < duration:
            applied_duration = maximum_duration
        else:
            applied_duration = duration
        self.charge_level += self.eta_d * power * applied_duration
        return power * applied_duration
        