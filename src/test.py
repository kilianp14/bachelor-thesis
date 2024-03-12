import pybamm
from netlist_utils import setup_circuit
from solvers import Manager

N_STEPS = 1000

netlist = setup_circuit(Np=4, Ns=1, Rb=1.5e-3, Rc=1e-2, Ri=5e-2, V=4.0, I=5.0)

parameter_values = pybamm.ParameterValues("Chen2020")

manager = Manager(
    netlist=netlist,
    parameter_values=parameter_values,
    initial_soc=0.5,
    step_size=5,
    Nsteps=N_STEPS,
)
for i in range(N_STEPS + 1):
    output = manager.perform_step(current=0.0)
print(output)
