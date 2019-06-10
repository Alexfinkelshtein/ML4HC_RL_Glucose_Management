#my first commit

# imports
from simglucose.simulation.user_interface import *
from DDPG.ddpg import main
# import .simglucose.simulation.user_interface as sim_inter

##
patient_names = get_patients([1, 2])
pump_name = get_insulin_pump(selection=2)
# pump = InsulinPump.withName(pump_name)
cgm_seed = 5
cgm_sensor_name = get_cgm_sensor(selection=1)

# sim_time = 24
# controller = BBController()
# start_time = '0'
# now = datetime.now()
# start_hour = timedelta(hours=float(0))
# start_time = datetime.combine(now.date(), datetime.min.time()) + start_hour
# scenario = CustomScenario(start_time=start_time, scenario=[(1, 300)])
# save_path = ''
# animate = True
# envs = our_build_envs(scenario, start_time, patient_names, cgm_sensor_name, cgm_seed, pump_name)
# env = envs[0]
import gym

# Register gym environment. By specifying kwargs,
# you are able to choose which patient to simulate.
# patient_name must be 'adolescent#001' to 'adolescent#010',
# or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
from gym.envs.registration import register
register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002'}
)

# env = gym.make('simglucose-adolescent2-v0')


args = {
    'env': 'simglucose-adolescent2-v0',
    'random_seed': 3,
    'actor_lr': 0.1,
    'tau': 0.3,
    'minibatch_size': 10,
    'critic_lr': 0.1,
    'gamma': 0.8,
    'use_gym_monitor': True,
    'render_env': True,
    'monitor_dir': r'C:\Users\afinkels\Desktop\private\Technion\Master studies\Machine Learning for Healthcare\project\ML4HC_RL_Glucose_Management\Results\Monitor',
    'buffer_size': 10,
    'summary_dir': r'C:\Users\afinkels\Desktop\private\Technion\Master studies\Machine Learning for Healthcare\project\ML4HC_RL_Glucose_Management\Results\Summaries',
    'max_episodes': 10,
    'max_episode_len': 60*24
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
