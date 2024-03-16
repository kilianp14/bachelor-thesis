import pybamm
from netlist_utils import setup_circuit
from battery import BatteryPack
from vessim.cosim import Environment
from vessim.controller import Monitor
from vessim.actor import ComputingSystem, Generator
from vessim.power_meter import MockPowerMeter
from vessim.signal import HistoricalSignal

netlist = setup_circuit(Np=4, Ns=1, Rb=1.5e-3, Rc=1e-2, Ri=5e-2, V=4.0, I=5.0)

parameter_values = pybamm.ParameterValues("Chen2020")

pack = BatteryPack(
    netlist=netlist,
    parameter_values=parameter_values,
    initial_soc=0.5,
    step_size=5,
)

environment = Environment(sim_start="2022-06-11 00:00:00")

monitor = Monitor()  # stores simulation result on each step
environment.add_microgrid(
    actors=[
        ComputingSystem(power_meters=[MockPowerMeter(p=1)]),
        Generator(signal=HistoricalSignal.from_dataset("solcast2022_global", params={"scale": 5.0}), column="Berlin"),
    ],
    controllers=[monitor],
    storage=pack,
    step_size=60,
)

environment.run(until=3600 * 24)
monitor.to_csv("result.csv")