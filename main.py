# imports
from simglucose.simulation.user_interface import *
from DDPG import ddpg
# import .simglucose.simulation.user_interface as sim_inter

##
patient_names = get_patients([1, 2])

pump = get_insulin_pump(selection=1)
cgm_seed = 5
cgm_sensor_name = get_cgm_sensor(selection=1)

sim_time = 24
controller = BBController()
start_time = 0
scenario = CustomScenario(start_time=start_time, scenario=[(1, 300)])
save_path = ''
animate = True
envs = our_build_envs(scenario, start_time, patient_names, cgm_sensor_name, cgm_seed, pump)
sim = SimObj(envs, controller, sim_time, animate=True, path=None)

# sim = create_sim_instance(sim_time=sim_time,
#                         scenario=scenario,
#                         controller=controller,
#                         start_time=start_time,
#                         save_path=save_path,
# #                         animate=True)

# simulate()
