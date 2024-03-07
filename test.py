import pybamm
import liionpack as lp

N_STEPS = 10

netlist = lp.setup_circuit(Np=4, Ns=1, Rb=1.5e-3, Rc=1e-2, Ri=5e-2, V=4.0, I=5.0)

parameter_values = pybamm.ParameterValues("Chen2020")

manager = lp.Manager(
    netlist=netlist,
    parameter_values=parameter_values,
    initial_soc=0.5,
    step_size=5,
    Nsteps=N_STEPS,
)
for i in range(N_STEPS + 1):
    output = manager.perform_step(current=6.0)
    print(output)
