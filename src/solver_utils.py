import casadi
import pybamm
import numpy as np
import pandas as pd
import pybamm
from typing import Optional


def serial_eval(model, solutions, inputs_dict, variables):
    """
    Internal function to evaluate the model variables in a serial way.

    Args:
        model (pybamm.Model):
            The built model
        solutions (iter of pybamm.Solution):
            Used to get the last state of the system and use as x0 and z0 for the
            casadi integrator. Provide solution objects for each battery.
        inputs_dict (iter of input_dicts):
            Provide inputs_dict objects for each battery.
        variables (variables evaluator):
            Produced by _create_casadi_objects when mapped = False

    Returns:
        sol (list):
            solutions that have been stepped forward by one timestep
        var_eval (list):
            evaluated variables for final state of system

    """
    len_rhs = model.concatenated_rhs.size
    N = len(solutions)
    t_min = 0.0
    var_eval = []
    for k in range(N):
        if solutions[k] is None:
            # First pass
            xend = model.y0[:len_rhs]
        else:
            xend = solutions[k].y[:, -1]

        temp = inputs_dict[k]
        inputs = casadi.vertcat(*[x for x in temp.values()] + [t_min])
        ninputs = len(temp.values())
        var_eval.append(variables(0, xend[:len_rhs], xend[len_rhs:], inputs[0:ninputs]))

    return casadi.horzcat(*var_eval)


def serial_step(model, solutions, inputs_dict, integrator, variables, t_eval, events):
    """
    Internal function to process the model for one timestep in a serial way.

    Args:
        model (pybamm.Model):
            The built model
        solutions (iter of pybamm.Solution):
            Used to get the last state of the system and use as x0 and z0 for the
            casadi integrator. Provide solution objects for each battery.
        inputs_dict (iter of input_dicts):
            Provide inputs_dict objects for each battery.
        integrator (casadi.integrator):
            Produced by _create_casadi_objects when mapped = False
        variables (variables evaluator):
            Produced by _create_casadi_objects when mapped = False
        t_eval (np.ndarray):
            A float array of times to evaluate.
            Produced by _create_casadi_objects when mapped = False
        events (mapped events evaluator):
            Produced by `_create_casadi_objects`

    Returns:
        sol (list):
            solutions that have been stepped forward by one timestep
        var_eval (list):
            evaluated variables for final state of system

    """
    len_rhs = model.concatenated_rhs.size
    N = len(solutions)
    t_min = 0.0
    timer = pybamm.Timer()
    sol = []
    var_eval = []
    events_eval = []
    for k in range(N):
        if solutions[k] is None:
            # First pass
            x0 = model.y0[:len_rhs]
            z0 = model.y0[len_rhs:]
        else:
            x0 = solutions[k].y[:len_rhs, -1]
            z0 = solutions[k].y[len_rhs:, -1]
        temp = inputs_dict[k]
        inputs = casadi.vertcat(*[x for x in temp.values()] + [t_min])
        ninputs = len(temp.values())
        # Call the integrator once, with the grid
        casadi_sol = integrator(x0=x0, z0=z0, p=inputs)
        xf = casadi.horzcat(x0, casadi_sol["xf"])
        zf = casadi_sol["zf"]
        if zf.is_empty():
            y_sol = xf
        else:
            y_sol = casadi.vertcat(xf, zf)
        xend = y_sol[:, -1]
        sol.append(pybamm.Solution(t_eval, y_sol, model, inputs_dict[k]))
        var_eval.append(variables(0, xend[:len_rhs], xend[len_rhs:], inputs[0:ninputs]))
        if events is not None:
            events_eval.append(
                events(0, xend[:len_rhs], xend[len_rhs:], inputs[0:ninputs])
            )
        integration_time = timer.time()
        sol[-1].integration_time = integration_time

    return sol, casadi.horzcat(*var_eval), casadi.horzcat(*events_eval)


def create_casadi_objects(inputs, sim, dt, variable_names):
    """
    Internal function to produce the casadi objects in their mapped form for
    parallel evaluation

    Args:
        inputs (dict):
            initial guess for inputs (not used for simulation).
        sim (pybamm.Simulation):
            A PyBaMM simulation object that contains the model, parameter values,
            solver, solution etc.
        dt (float):
            The time interval (in seconds) for a single timestep. Fixed throughout
            the simulation
        variable_names (list):
            Variables to evaluate during solve. Must be a valid key in the
            model.variables

    Returns:
        integrator (mapped casadi.integrator):
            Solves an initial value problem (IVP) coupled to a terminal value
            problem with differential equation given as an implicit ODE coupled
            to an algebraic equation and a set of quadratures
        variables_fn (mapped variables evaluator):
            evaluates the simulation and output variables. see casadi function
        t_eval (np.ndarray):
            Float array of times to evaluate.
            times to evaluate in a single step, starting at zero for each step
        events_fn (mapped events evaluator):
            evaluates the event variables. see casadi function

    """
    solver = sim.solver
    # Initial solution - this builds the model behind the scenes
    initial_solutions = []
    init_sol = sim.step(
        dt=1e-6, save=False, starting_solution=None, inputs=inputs[0]
    ).last_state
    # evaluate initial condition
    model = sim.built_model
    y0_total_size = (
        model.len_rhs + model.len_rhs_sens + model.len_alg + model.len_alg_sens
    )
    y_zero = np.zeros((y0_total_size, 1))
    for inpt in inputs:
        inputs_casadi = casadi.vertcat(*[x for x in inpt.values()])
        initial_solutions.append(init_sol.copy())
        _init = model.initial_conditions_eval(0, y_zero, inputs_casadi)
        initial_solutions[-1].y[:] = _init

    # Step model forward dt seconds
    t_eval = np.linspace(0, dt, 11)

    # No external variables - Temperature solved as lumped model in pybamm
    # External variables could (and should) be used if battery thermal problem
    # Includes conduction with any other circuits or neighboring batteries
    # inp_and_ext.update(external_variables)
    inp_and_ext = inputs

    # Code to create mapped integrator
    integrator = solver.create_integrator(
        sim.built_model, inputs=inp_and_ext, t_eval=t_eval
    )
    # Get the input parameter order
    ip_order = inputs[0].keys()
    # Variables function for parallel evaluation
    casadi_objs = sim.built_model.export_casadi_objects(
        variable_names=variable_names, input_parameter_order=ip_order
    )
    variables = casadi_objs["variables"]
    t, x, z, p = (
        casadi_objs["t"],
        casadi_objs["x"],
        casadi_objs["z"],
        casadi_objs["inputs"],
    )
    variables_stacked = casadi.vertcat(*variables.values())
    variables_fn = casadi.Function("variables", [t, x, z, p], [variables_stacked])
    # Look for events in model variables and create a function to evaluate them
    all_vars = sorted(sim.model.variables.keys())
    event_vars = [v for v in all_vars if "Event" in v]
    if len(event_vars) > 0:
        # Variables function for parallel evaluation
        casadi_objs = sim.built_model.export_casadi_objects(
            variable_names=variable_names, input_parameter_order=ip_order
        )
        events = casadi_objs["variables"]
        t, x, z, p = (
            casadi_objs["t"],
            casadi_objs["x"],
            casadi_objs["z"],
            casadi_objs["inputs"],
        )
        events_stacked = casadi.vertcat(*events.values())
        events_fn = casadi.Function("variables", [t, x, z, p], [events_stacked])
    else:
        events_fn = None

    output = {
        "integrator": integrator,
        "variables_fn": variables_fn,
        "t_eval": t_eval,
        "event_names": event_vars,
        "events_fn": events_fn,
        "initial_solutions": initial_solutions,
    }
    return output


def build_inputs_dict(I_batt):
    """
    Function to convert inputs and external_variable arrays to list of dicts
    As expected by the casadi solver. These are then converted back for mapped
    solving but stored individually on each returned solution.
    Can probably remove this process later

    Args:
        I_batt (np.ndarray):
            The input current for each battery.

    Returns:
        inputs_dict (list):
            each element of the list is an inputs dictionary corresponding to each
            battery.
    """
    dicts = []
    for i_batt in I_batt:
        dicts.append({"Current function [A]": i_batt})
    return dicts


def setup_basic_simulation(
    model: pybamm.lithium_ion.BaseModel,
    parameter_values: pybamm.ParameterValues,
    initial_soc: float = 0,
    geometry: Optional[pybamm.Geometry] = None,
    submesh_types: Optional[dict] = None,
    var_pts: Optional[dict] = None,
    spatial_methods: Optional[dict] = None,
    solver: Optional[pybamm.BaseSolver] = None,
):
    # Get data for state-of-charge estimation
    df = pd.read_csv("data/SoC_Values.csv", index_col=0)
    ocv_values = df["ocv_values"].to_numpy()
    soc_values = df["soc_values"].to_numpy()
    #ocv_values = [parameter_values["Lower voltage cut-off [V]"]]
    #soc_values = [0.0]
    #for i in np.arange(0.01, 1, 0.001):
        #soc_sim = pybamm.Simulation(
            #model=model,
            #geometry=geometry,
            #parameter_values=parameter_values,
            #submesh_types=submesh_types,
            #var_pts=var_pts,
            #spatial_methods=spatial_methods,
            #solver=solver,
        #)
        #soc_sim.build(initial_soc=i)
        #if parameter_values["Current function [A]"].__class__ is pybamm.InputParameter:
            #sol = soc_sim.step(dt=1e-6, inputs={"Current function [A]": 0.0}).last_state
        #else:
            #sol = soc_sim.step(dt=1e-6, inputs={"Power function [W]": 0.0}).last_state
        #soc_values.append(i)
        #ocv_values.append(sol["Battery open-circuit voltage [V]"].data[-1])
    #ocv_values.append(parameter_values["Upper voltage cut-off [V]"])
    #soc_values.append(1.0)
    #df = pd.DataFrame({'ocv_values': ocv_values, 'soc_values': soc_values})
    #df.to_csv('data/SoC_Values.csv', index=1)

    sim = pybamm.Simulation(
        model=model,
        geometry=geometry,
        parameter_values=parameter_values,
        submesh_types=submesh_types,
        var_pts=var_pts,
        spatial_methods=spatial_methods,
        solver=solver,
    )
    # Bugs sometimes if initial soc is 0 or 1
    if initial_soc > 0.999:
        initial_soc = 0.999
    if initial_soc < 0.001:
        initial_soc = 0.001
    sim.build(initial_soc=initial_soc)
    return sim, ocv_values, soc_values