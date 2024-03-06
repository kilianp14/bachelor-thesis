import pybamm
import liionpack as lp

netlist = lp.setup_circuit(Np=4, Ns=1, Rb=1.5e-3, Rc=1e-2, Ri=5e-2, V=4.0, I=5.0)

parameter_values = pybamm.ParameterValues("Chen2020")

output = lp.solve(
    netlist=netlist,
    parameter_values=parameter_values,
    initial_soc=0.5,
)
