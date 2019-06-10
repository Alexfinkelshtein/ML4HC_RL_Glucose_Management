#my first commit

# imports
from simglucose.simulation.user_interface import *
from DDPG.ddpg import main
# import .simglucose.simulation.user_interface as sim_inter

##
patient_names = get_patients([1, 2])

pump = get_insulin_pump(selection=1)
cgm_seed = 5
cgm_sensor_name = get_cgm_sensor(selection=1)

sim_time = 24
controller = BBController()
start_time = '0'
now = datetime.now()
start_hour = timedelta(hours=float(0))
start_time = datetime.combine(now.date(), datetime.min.time()) + start_hour
scenario = CustomScenario(start_time=start_time, scenario=[(1, 300)])
save_path = ''
animate = True
envs = our_build_envs(scenario, start_time, patient_names, cgm_sensor_name, cgm_seed, pump)
env = envs[0]
args = {
    'env': env,
    'random_seed': 3,
    'actor_lr': 0.1,
    'tau': 0.3,
    'minibatch_size': 10,
    'critic_lr': 0.1,
    'gamma': 0.8,
    'use_gym_monitor': False,
    'render_env': False,
    'monitor_dir': '',
    'buffer_size': 10,
    'summary_dir': r'C:\Users\afinkels\Desktop\private\Technion\Master studies\Machine Learning for Healthcare\project\ML4HC_RL_Glucose_Management\Results\Summaries',
    'max_episodes': 10,
    'max_episode_len': 24
}

main(args)

# sim = SimObj(envs, controller, sim_time, animate=True, path=None)
#
# sim = create_sim_instance(sim_time=sim_time,
#                         scenario=scenario,
#                         controller=controller,
#                         start_time=start_time,
#                         save_path=save_path,
# #                         animate=True)
#
# # simulate()
