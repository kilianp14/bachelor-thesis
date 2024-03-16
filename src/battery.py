import numpy as np
import pybamm

from vessim.storage import Storage

from solver_utils import create_casadi_objects, serial_step, serial_eval, setup_basic_simulation, build_inputs_dict
from netlist_utils import solve_circuit_vectorized, power_loss


class Actor:
    def __init__(self):
        pass

    def setup(
        self,
        parameter_values,
        dt,
        inputs,
        variable_names,
        initial_soc,
    ):
        # Set up simulation
        self.simulation = setup_basic_simulation(parameter_values, initial_soc)

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


class BatteryPack(Storage):
    """ 
    Class for step-by-step solving of a lithium-ion-battery-pack.

    Args:
        netlist (pandas.DataFrame):
            A netlist of circuit elements with format. desc, node1, node2, value.
            Produced by liionpack.read_netlist or liionpack.setup_circuit
        parameter_values (pybamm.ParameterValues):
            A dictionary of all the model parameters
        inputs (dict):
            Dictionary for every model input with value for each battery
        initial_soc (float):
            The initial state of charge for every battery. The default is None
            in which case concentrations set in the parameter_values are used.
        output_variables (list):
            Variables to evaluate during solve. Must be a valid key in the
            model.variables
    """
    def __init__(
        self,
        netlist,
        parameter_values,
        step_size,
        initial_soc,
        output_variables = None,
    ):
        self.netlist = netlist
        self.parameter_values = parameter_values
        self.check_current_function()
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
        self.last_current = 0.0
        self.V_terminal = np.float32(V_node[self.Terminal_Node][0])

        self.v_cut_lower = parameter_values["Lower voltage cut-off [V]"]
        self.v_cut_higher = parameter_values["Upper voltage cut-off [V]"]

        self.inputs_dict = build_inputs_dict(self.shm_i_app)
        self.actor = Actor()
        self.actor.setup(
            parameter_values=self.parameter_values,
            dt=self.step_size,
            inputs=self.inputs_dict,
            variable_names=self.variable_names,
            initial_soc=initial_soc,
        )
        # Get the initial state of the system
        self.actor.evaluate(self.inputs_dict)
        self.step = -1
    
    def state(self):
        state = {}
        state["Pack current [A]"] = self.last_current
        state["Pack terminal voltage [V]"] = self.V_terminal
        state["Cell current [A]"] = self.shm_i_app[:]
        state["Cell internal resistance [Ohm]"] = self.shm_Ri[:]
        for j in range(self.Nvar):
            state[self.variable_names[j]] = self.output[j,:]
        return state
    
    def soc(self):
        pass

    def update(self, power, duration):
        current = power / self.V_terminal
        self.perform_step(current)
        return 0.0

    def perform_step(self, current):
        self.step += 1
        # 01 Calculate whether resting or restarting
        self.resting = (
            self.step > 0 and current == 0.0 and self.last_current == 0.0
        )
        self.restarting = (
            self.step > 0 and current != 0.0 and self.last_current == 0.0
        )
        # 02 Get the actor output - Battery state info
        self.output = self.actor.output()
        # 03 Get the ocv and internal resistance
        temp_v = self.output[0,:]
        temp_ocv = self.output[1,:]
        # When resting and rebalancing currents are small the internal
        # resistance calculation can diverge as it's R = V / I
        # At rest the internal resistance should not change greatly
        # so for now just don't recalculate it.
        if not self.resting and not self.restarting:
            self.temp_Ri = self.calculate_internal_resistance()
        self.shm_Ri[:] = self.temp_Ri
        # 04 Update netlist
        self.netlist.loc[self.V_map, ("value")] = temp_ocv
        self.netlist.loc[self.Ri_map, ("value")] = self.temp_Ri
        self.netlist.loc[self.I_map, ("value")] = current
        power_loss(self.netlist)
        # 05 Solve the circuit with updated netlist
        V_node, I_batt = solve_circuit_vectorized(self.netlist)
        self.record_times = self.step * self.step_size
        self.V_terminal = np.float32(V_node[self.Terminal_Node][0])
        I_app = I_batt[:] * -1
        self.shm_i_app[:] = I_app.astype(np.float32)
        self.inputs_dict = build_inputs_dict(I_app)
        # 06 Check if voltage limits are reached and terminate
        if np.any(temp_v < self.v_cut_lower):
            raise RuntimeError("Low voltage limit reached")
        if np.any(temp_v > self.v_cut_higher):
            raise RuntimeError("High voltage limit reached")
        # 07 Step the electrochemical system
        events = self.actor.step(self.inputs_dict)
        if events:
            self.log_event()
        
        self.shm_Ri = np.abs(self.shm_Ri)

        self.last_current = current

    def check_current_function(self):
        i_func = self.parameter_values["Current function [A]"]
        if i_func.__class__ is not pybamm.InputParameter:
            self.parameter_values.update({"Current function [A]": "[input]"})

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
