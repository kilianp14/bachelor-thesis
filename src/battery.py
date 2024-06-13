from __future__ import annotations
from typing import Optional

import pybamm
import numpy as np
import vessim as vs

from solver_utils import create_casadi_objects, serial_step, serial_eval, setup_basic_simulation, build_inputs_dict
from netlist_utils import solve_circuit_vectorized, power_loss


class CLCBattery(vs.Storage):
    """CLC Battery model for lithium-ion batteries. Default is the LGM50 21700 parameterization.

    Args:
        cell_capacity: Single cell battery capacity in Wh. Default is 18.2Wh.
        initial_soc: Initial battery state-of-charge. Has to be between 0 and 1. Defaults to 0.
        nom_voltage: Single cell nominal voltage in V. Defaults to 3.63V.
        alpha_d: Maximum discharging C-rate. Defaults to 3.0C.
        alpha_c: Maximum charging C-rate. Defaults to 0.7C.
        eta_d: Average fraction of power that has to be discharged from battery to obtain said
            power. Is equivalent to the discharging inefficiency. Defaults to 1.043.
        eta_c: Average fraction of power that is stored in battery when charged at said power.
            Is equivalent to the charging inefficiency. Defualts to 0.979.
        u_1: Linear factor for the lower state-of-charge limit depending on the applied discharge
            rate. Defaults to 0.069.
        v_1: Offset for the lower state-of-charge limit depending on the applied discharge rate.
            Defaults to 0.0.
        u_2: Linear factor for the upper state-of-charge limit depending on the applied charge
            rate. Defaults to -0.081.
        v_2: Offset for the upper state-of-charge limit depending on the applied charge rate.
            Defaults to 1.
    """
    def __init__(
        self,
        number_of_cells: int = 1,
        cell_capacity: float = 18.2,
        initial_soc: float = 0,
        min_soc: float = 0,
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
        assert number_of_cells > 0, "There has to be a positive number of cells."
        self.number_of_cells = number_of_cells
        self.cell_capacity = cell_capacity # Wh
        assert 0 <= initial_soc <= 1, "Invalid initial state-of-charge. Has to be between 0 and 1."
        self.charge_level = cell_capacity * initial_soc # Wh
        self.min_soc = min_soc
        self.nom_voltage = nom_voltage # V
        capacity = self.cell_capacity / self.nom_voltage # A
        self.alpha_d = - alpha_d * capacity # C-Rate to A
        self.alpha_c = alpha_c * capacity # C-Rate to A
        self.eta_d = eta_d
        self.eta_c = eta_c
        self.u_1 = u_1 * self.nom_voltage # C-Rate at SoC -> A at Wh
        self.v_1 = v_1 * self.cell_capacity # C-Rate at SoC -> A at Wh
        self.u_2 = u_2 * self.nom_voltage # C-Rate at SoC -> A at Wh
        self.v_2 = v_2 * self.cell_capacity # C-Rate at SoC -> A at Wh

    def soc(self) -> float:
        return self.charge_level / self.cell_capacity

    def update(self, power: float, duration: int) -> float:
        applied_power = power
        current = applied_power/ self.nom_voltage
        if current < self.alpha_d * self.number_of_cells:
            print("Discharging current exceeds maximum discharge rate.")
            current = self.alpha_d * self.number_of_cells
            applied_power = current * self.nom_voltage
        elif current > self.alpha_c * self.number_of_cells:
            print("Charging current exceeds maximum charging rate.")
            current = self.alpha_c * self.number_of_cells
            applied_power = current * self.nom_voltage

        if applied_power > 0:
            return self.charge(applied_power, current, duration)
        elif applied_power < 0:
            return self.discharge(applied_power, current, duration)
        else:
            return 0

    def charge(self, power: float, current: float, duration: int) -> float:
        energy_limit = self.u_2 * current / self.number_of_cells + self.v_2
        maximum_duration = (
            (energy_limit - self.charge_level) * self.number_of_cells * 3600 / (self.eta_c * power)
        )
        if maximum_duration < 0:
            return 0
        elif maximum_duration < duration:
            applied_duration = maximum_duration
        else:
            applied_duration = duration
        self.charge_level += self.eta_c * power * applied_duration / (self.number_of_cells * 3600)
        return power * applied_duration

    def discharge(self, power: float, current: float, duration: int) -> float:
        energy_limit = np.maximum(
            self.u_1 * (-current) / self.number_of_cells + self.v_1,
            self.min_soc * self.cell_capacity,
        )
        maximum_duration = (
            (energy_limit - self.charge_level) * self.number_of_cells * 3600 / (self.eta_d * power)
        )
        if maximum_duration < 0:
            return 0
        elif maximum_duration < duration:
            applied_duration = maximum_duration
        else:
            applied_duration = duration
        self.charge_level += self.eta_d * power * applied_duration / (self.number_of_cells * 3600)
        return power * applied_duration

    def state(self) -> dict:
        return {
            "soc": self.soc(),
            "charge_level": self.charge_level * self.number_of_cells,
            "capacity": self.cell_capacity * self.number_of_cells,
            "min_soc": self.min_soc,
        }


class PybammBattery(vs.Storage):
    def __init__(
        self,
        model: pybamm.lithium_ion.BaseModel,
        initial_soc: float = 0,
        number_of_cells: int = 1,
        geometry: Optional[pybamm.Geometry] = None,
        parameter_values: Optional[pybamm.ParameterValues] = None,
        submesh_types: Optional[dict] = None,
        var_pts: Optional[dict] = None,
        spatial_methods: Optional[dict] = None,
        solver: Optional[pybamm.BaseSolver] = None,
        output_variables: Optional[list] = None,
    ) -> None:
        parameter_values = parameter_values if parameter_values else model.default_parameter_values
        parameter_values.update({"Power function [W]": "[input]"}, check_already_exists=False)
        assert number_of_cells > 0, "There has to be a positive number of cells."
        self.number_of_cells = number_of_cells

        self.variable_names: list[str] = []
        if output_variables is not None:
            for out in output_variables:
                if out not in self.variable_names:
                    self.variable_names.append(out)

        self.sim, self._ocv_values, self._soc_values = setup_basic_simulation(
            model, parameter_values, initial_soc=initial_soc, geometry=geometry, submesh_types=submesh_types, var_pts=var_pts, spatial_methods=spatial_methods, solver=solver
        )

        # Get initial solution
        self.cell_solution = self.sim.step(dt=1e-6, inputs={"Power function [W]": 0.0}).last_state

    def soc(self) -> float:
        value = self.cell_solution['Surface open-circuit voltage [V]'].data[0]
        idx = np.searchsorted(self._ocv_values, value, side='right')
        if idx == 0:
            return 0.0
        elif idx == self._ocv_values.size:
            return 1.0
        else:
            x1, x2 = self._ocv_values[idx - 1], self._ocv_values[idx]
            y1, y2 = self._soc_values[idx - 1], self._soc_values[idx]
            return y1 + (value - x1) * (y2 - y1) / (x2 - x1)

    def update(self, power: float, duration: int) -> float:
        self.cell_solution = self.sim.step(duration, inputs={"Power function [W]": - power / self.number_of_cells}).last_state
        return power * duration

    def state(self) -> dict:
        state = {"soc": self.soc()}
        for variable in self.variable_names:
            state[f"single_cell_{variable}"] = self.cell_solution[variable].data[0]
        return state


class LiionBatteryPack(vs.Storage):
    def __init__(
        self,
        model: pybamm.lithium_ion.BaseModel,
        netlist,
        step_size,
        initial_soc: float = 0,
        geometry: Optional[pybamm.Geometry] = None,
        parameter_values: Optional[pybamm.ParameterValues] = None,
        submesh_types: Optional[dict] = None,
        var_pts: Optional[dict] = None,
        spatial_methods: Optional[dict] = None,
        output_variables: Optional[list] = None,
    ) -> None:
        parameter_values = parameter_values if parameter_values else model.default_parameter_values
        parameter_values.update({"Current function [A]": "[input]"}, check_already_exists=False)
        self.v_cut_lower = parameter_values["Lower voltage cut-off [V]"]
        self.v_cut_higher = parameter_values["Upper voltage cut-off [V]"]

        self.netlist = netlist

        # Get netlist indices for resistors, voltage sources, current sources
        self.Ri_map = netlist["desc"].str.find("Ri") > -1
        self.V_map = netlist["desc"].str.find("V") > -1
        self.I_map = netlist["desc"].str.find("I") > -1
        self.Terminal_Node = np.array(netlist[self.I_map].node1)
        self.Nspm = np.sum(self.V_map)

        self.step_size = step_size
        # If the step is starting with a rest the current will be zero and
        # this messes up the internal resistance calc. Add a very small current
        # for init.
        netlist.loc[self.I_map, ("value")] = 1e-3
        # Solve the circuit to initialise the electrochemical models
        V_node, I_batt = solve_circuit_vectorized(netlist)

        # The simulation output variables calculated at each step for each battery
        # Must be a 0D variable i.e. battery wide volume average - or X-averaged for
        # 1D model
        self.variable_names = [
            "Terminal voltage [V]",
            "Surface open-circuit voltage [V]",
        ]
        if output_variables is not None:
            for out in output_variables:
                if out not in self.variable_names:
                    self.variable_names.append(out)
        self.Nvar = len(self.variable_names)

        # Storage variables for simulation data
        self.shm_Ri = np.zeros([self.Nspm], dtype=np.float32)
        self.output = np.zeros([self.Nvar, self.Nspm], dtype=np.float32)
        self.shm_i_app = (I_batt * -1).astype(np.float32)
        self.last_power = 0.0
        self.V_terminal = np.float32(V_node[self.Terminal_Node][0])

        self.inputs_dict = build_inputs_dict(self.shm_i_app)
        sim, self._ocv_values, self._soc_values = setup_basic_simulation(
            model, parameter_values, initial_soc=initial_soc, geometry=geometry, submesh_types=submesh_types, var_pts=var_pts, spatial_methods=spatial_methods, solver=pybamm.CasadiSolver(mode="safe")
        )
        self.actor = Actor()
        self.actor.setup(
            dt=self.step_size,
            inputs=self.inputs_dict,
            variable_names=self.variable_names,
            simulation=sim,
        )
        # Get the initial state of the system
        self.actor.evaluate(self.inputs_dict)
        self.output = self.actor.output()
        self.last_power = -1.0

    def soc(self):
        soc = 0
        for i in range(self.Nspm):
            value = self.output[1,i]
            idx = np.searchsorted(self._ocv_values, value, side='right')
            if idx == 0:
                soc += 0.0
            elif idx == self._ocv_values.size:
                soc += 1.0
            else:
                x1, x2 = self._ocv_values[idx - 1], self._ocv_values[idx]
                y1, y2 = self._soc_values[idx - 1], self._soc_values[idx]
                soc += y1 + (value - x1) * (y2 - y1) / (x2 - x1)
        return soc / self.Nspm
    
    def state(self):
        state = {}
        state["soc"] = self.soc()
        state["pack_terminal_voltage"] = self.V_terminal
        state["cell_current"] = {f"battery_{index}": value for index, value in enumerate(self.shm_i_app)}
        state["cell_internal_resistance"] = {f"battery_{index}": value for index, value in enumerate(self.shm_Ri)}
        for j in range(self.Nvar):
            state[self.variable_names[j]] = {f"battery_{index}": value for index, value in enumerate(self.output[j,:])}
        return state
    
    def update(self, power, duration):
        # Calculate whether resting or restarting
        self.resting = power == 0.0 and self.last_power == 0.0
        self.restarting = power != 0.0 and self.last_power == 0.0
        # Get the ocv and internal resistance
        temp_v = self.output[0,:]
        temp_ocv = self.output[1,:]
        # When resting and rebalancing currents are small the internal
        # resistance calculation can diverge as it's R = V / I
        # At rest the internal resistance should not change greatly
        # so for now just don't recalculate it.
        if not self.resting and not self.restarting:
            self.temp_Ri = self.calculate_internal_resistance()
        self.shm_Ri[:] = self.temp_Ri
        # Update netlist
        self.netlist.loc[self.V_map, ("value")] = temp_ocv
        self.netlist.loc[self.Ri_map, ("value")] = self.temp_Ri
        # Solve the circuit
        power_loss(self.netlist, include_Ri=True)
        V_node, I_batt = solve_circuit_vectorized(self.netlist, -power)
        self.V_terminal = np.float32(V_node[self.Terminal_Node][0])
        I_app = I_batt[:] * -1
        self.shm_i_app[:] = I_app.astype(np.float32)
        self.inputs_dict = build_inputs_dict(I_app)
        # Check if voltage limits are reached and terminate
        if np.any(temp_v < self.v_cut_lower):
            raise RuntimeError("Low voltage limit reached")
        if np.any(temp_v > self.v_cut_higher):
            raise RuntimeError("High voltage limit reached")
        # Step the electrochemical system
        events = self.actor.step(self.inputs_dict)
        if events:
            self.log_event()
        
        self.shm_Ri = np.abs(self.shm_Ri)

        # Get the actor output - Battery state info
        self.output = self.actor.output()

        self.last_power = power
        return power * self.step_size

    def actor_htc(self, index):
        return self.htc[index]

    def calculate_internal_resistance(self):
        # Calculate internal resistance and update netlist
        temp_v = self.output[0,:]
        temp_ocv = self.output[1,:]
        temp_I = self.shm_i_app[:]
        temp_Ri = np.abs((temp_ocv - temp_v) / temp_I)
        temp_Ri[temp_Ri == 0.0] = 1e-6
        return temp_Ri

    def log_event(self):
        event_change = np.asarray(self.actor.get_event_change())
        Nr, Nc = event_change.shape
        event_names = self.actor.get_event_names()
        for r in range(Nr):
            if np.any(event_change[r, :]):
                print(
                    event_names[r]
                    + ", Batteries: "
                    + str(np.where(event_change[r, :])[0].tolist())
                )


class Actor:
    def __init__(self):
        pass

    def setup(
        self,
        dt,
        inputs,
        variable_names,
        simulation
    ):
        # Set up simulation
        self.simulation = simulation

        # Set up integrator
        casadi_objs = create_casadi_objects(inputs, self.simulation, dt, variable_names)
        self.model = self.simulation.built_model
        self.integrator = casadi_objs["integrator"]
        self.variables_fn = casadi_objs["variables_fn"]
        self.t_eval = casadi_objs["t_eval"]
        self.event_names = casadi_objs["event_names"]
        self.events_fn = casadi_objs["events_fn"]
        self.step_solutions = casadi_objs["initial_solutions"]
        self.last_events = None
        self.event_change = None

    def step(self, inputs):
        # Solver Step
        self.step_solutions, self.var_eval, self.events_eval = serial_step(
            self.simulation.built_model,
            self.step_solutions,
            inputs,
            self.integrator,
            self.variables_fn,
            self.t_eval,
            self.events_fn,
        )
        return self.check_events()

    def evaluate(self, inputs):
        self.var_eval = serial_eval(
            self.simulation.built_model,
            self.step_solutions,
            inputs,
            self.variables_fn,
        )

    def check_events(self):
        if self.last_events is not None:
            # Compare changes
            new_sign = np.sign(self.events_eval)
            old_sign = np.sign(self.last_events)
            self.event_change = (old_sign * new_sign) < 0
            self.last_events = self.events_eval
            return np.any(self.event_change)
        else:
            self.last_events = self.events_eval
            return False

    def get_event_change(self):
        return self.event_change

    def get_event_names(self):
        return self.event_names

    def output(self):
        return np.array(self.var_eval, dtype=np.float32)