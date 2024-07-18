import pybamm
from netlist_utils import setup_circuit
from battery import LiionBatteryPack, PybammBattery, CLCBattery
import vessim as vs
import timeit
import sys
import csv
import gc


STEP_SIZE = 1

def timed_experiment(environment, number_of_steps):
    environment.run(until=number_of_steps)

def setup_experiment(battery_type, number_of_cells):
    gc.collect()
    if battery_type == "SimpleBattery":
        battery = vs.SimpleBattery(capacity=18.87 * number_of_cells * number_of_cells, initial_soc=1)
    elif battery_type == "CLCBattery":
        battery = CLCBattery(number_of_cells=number_of_cells * number_of_cells, initial_soc=1)
    elif battery_type == "PybammBattery":
        battery = PybammBattery(
            model_type=pybamm.lithium_ion.SPMe,
            initial_soc=1,
            number_of_cells=number_of_cells * number_of_cells,
            parameter_values=pybamm.ParameterValues("Chen2020"),
            output_variables=["Current [A]", "Voltage [V]", "Power [W]", "Discharge energy [W.h]"],
        )
    elif battery_type == "LiionBatteryPack":
        battery = LiionBatteryPack(
            model_type=pybamm.lithium_ion.SPMe,
            netlist=setup_circuit(Np=number_of_cells, Ns=number_of_cells, I=0.0),
            step_size=STEP_SIZE,
            initial_soc=1,
            parameter_values=pybamm.ParameterValues("Chen2020"),
        )
    power = -18.87 * 0.2 * number_of_cells * number_of_cells
    environment = vs.Environment(sim_start="2022-07-03 09:00:00")
    environment.add_microgrid(
        actors=[vs.Actor(name=f"MockActor:{type(battery).__name__}", signal=vs.MockSignal(value=power))],
        storage=battery,
        step_size=STEP_SIZE,
    )
    return environment

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Too few arguments")
    else:
        battery_type = sys.argv[1]
        number_of_cells = int(sys.argv[2])
        number_of_steps = int(sys.argv[3]) * STEP_SIZE
        number_of_repeats = int(sys.argv[4])

        results = timeit.repeat(
            stmt=f'timed_experiment(environment, {number_of_steps})',
            setup=f"environment = setup_experiment('{battery_type}', {number_of_cells})",
            repeat=number_of_repeats,
            number=1,
            globals=globals()
        )

        with open(f"results/Times_{battery_type}_{number_of_cells}.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            for result in results:
                writer.writerow([result / number_of_steps])
