from simglucose.simulation.user_interface import *

#
patient_names = our_pick_patients([1, 2])
cgm_sensor_name, cgm_seed = pick_cgm_sensor()
insulin_pump_name = pick_insulin_pump()

sim_time = 24
controller = BBController()
start_time = 0
scenario = CustomScenario(start_time=start_time, scenario=[(1, 300)])
save_path = ''
animate = True
envs = our_build_envs(scenario, start_time, patient_names, cgm_sensor_name, cgm_seed, insulin_pump_name)
sim = SimObj(envs, controller, sim_time, animate=True, path=None)
# sim = create_sim_instance(sim_time=sim_time,
#                         scenario=scenario,
#                         controller=controller,
#                         start_time=start_time,
#                         save_path=save_path,
# #                         animate=True)

# simulate()
