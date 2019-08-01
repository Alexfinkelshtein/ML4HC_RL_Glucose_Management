from DDPG.ddpg import main
from gym.envs.registration import register
import os
import datetime
import numpy as np
# imports
from simglucose.simulation.user_interface import *
import gym
import logging
import json
# import .simglucose.simulation.user_interface as sim_inter

##
# patient_names = get_patients([1, 2])
# pump_name = get_insulin_pump(selection=2)
# # pump = InsulinPump.withName(pump_name)
# cgm_seed = 5
# cgm_sensor_name = get_cgm_sensor(selection=1)

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

# Register gym environment. By specifying kwargs,
# you are able to choose which patient to simulate.
# patient_name must be 'adolescent#001' to 'adolescent#010',
# or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
# def reward(cgm, hyper_thresh=180, hypo_thresh=70):
#     if cgm > hyper_thresh:
#         return -1
#     if cgm < hypo_thresh:
#         return -5
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


register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002'}#, 'reward_fun': reward}
)

env = gym.make('simglucose-adolescent2-v0')

def summary_path(summaries_base):
    rel_files = np.array(os.listdir(summaries_base))
    files = list(map(lambda x: os.path.join(summaries_base, x), rel_files))
    rel_dirs = rel_files[list(map(os.path.isdir, files))]

    if len(rel_dirs) == 0:
        new_dir = os.path.join(summaries_base, '0')
    else:
        def my_value(x):  # if not number -1
            try:
                return float(x)
            except:
                return -1
        rel_dirs = sorted(rel_dirs, key=my_value, reverse=True)
        new_dir = os.path.join(summaries_base, str(float(rel_dirs[0]) + 1))
    os.makedirs(new_dir)
    return new_dir


summaries_base = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\Machine Learning for Healthcare\project\ML4HC_RL_Glucose_Management\Results\Summaries'
current_summary = summary_path(summaries_base)
logging.basicConfig(filename=os.path.join(current_summary, 'log.log'), level=logging.INFO)

args = {
    'env': 'simglucose-adolescent2-v0',
    'random_seed': 3,
    'actor_lr': 0.01,
    'tau': 0.3,
    'minibatch_size': 50,  # horizon??
    'critic_lr': 0.01,
    'gamma': 0.8,
    'use_gym_monitor': True,
    'render_env': False,  # plot episodes
    'monitor_dir': r'C:\Users\afinkels\Desktop\private\Technion\Master studies\Machine Learning for Healthcare\project\ML4HC_RL_Glucose_Management\Results\Monitor',
    'buffer_size': 100,
    'summary_dir': current_summary,
    'max_episodes': 500,
    'max_episode_len': 60*24
}
current_time = str(datetime.now())
logging.info(f'Start Timestamp: {current_time}')
args4print = json.dumps(args, sort_keys=True, indent=4)
logging.info(f'Arguments:\n {args4print}')
main(args)

