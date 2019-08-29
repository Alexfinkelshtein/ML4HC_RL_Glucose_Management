from DDPG.ddpg import train_ddpg, DDPG_Controller
from gym.envs.registration import register
import os
import datetime
import numpy as np
# imports
from simglucose.simulation.user_interface import *
import gym
import logging
import json
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# %% functions

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
        new_dir = os.path.join(summaries_base, str(int(rel_dirs[0]) + 1))
    os.makedirs(new_dir)
    return new_dir


def random_meal(time_min=0, time_max=20, size_min=5, size_max=70):
    time = np.random.choice(range(time_min, time_max))
    size = np.random.choice(range(size_min, size_max))
    return (time, size)

def random_meals(num_meals=3, time_min=0, time_max=20, size_min=5, size_max=70):
    times = []
    sizes = []
    meals = []
    for i in range(num_meals):
        (time, size) = random_meal(time_min=0, time_max=20, size_min=5, size_max=70)
        if time not in times:  # avoid having two meals at same time
            meals.append((time, size))
    return meals


if __name__ == "__main__":
    # %% CONFIG\PATHS\LOGGER
    mode = 'train'
    # mode = 'inference'

    animate = False  # control simglucose graphic animations
    patient_number = '2'
    patient_name = f'adolescent#00{patient_number}'

    summaries_base = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\Machine Learning for Healthcare\project\ML4HC_RL_Glucose_Management\Results\Summaries'
    monitor_dir = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\Machine Learning for Healthcare\project\ML4HC_RL_Glucose_Management\Results\Monitor'
    current_summary_path = summary_path(summaries_base)
    logging.basicConfig(filename=os.path.join(current_summary_path, 'log.log'), level=logging.INFO)

    # %% create scenario
    now = datetime.now()
    start_hour = timedelta(hours=float(0))
    start_time = datetime.combine(now.date(), datetime.min.time()) + start_hour
    # meals_ = random_meals(num_meals=4)  # example for randomized scenario creation
    meals = [(1, 60), (3, 80), (5, 60)]  # format: list of tuples, where (meal_time, meal_size [grams])
    scenario = CustomScenario(start_time=start_time, scenario=meals)
    # %% REGISTER GYM ENVIRONMENT

    # acceptable parameters at simglucose_gym_env
    register(id='simglucose-adolescent2-v0',
             entry_point='simglucose.envs:T1DSimEnv',
             kwargs={'patient_name': patient_name, 'custom_scenario': scenario, 'animate': animate}
             )
    env = gym.make('simglucose-adolescent2-v0')

    # %% DDPG Controller
    sensor_sample_time = 3
    load_model = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\Machine Learning for Healthcare\project\ML4HC_RL_Glucose_Management\Results\Summaries\166'
    args = {
        'env': f'simglucose-adolescent{patient_number}-v0',
        'random_seed': 1234,
        'actor_lr': 0.0005,
        'tau': 0.01,
        'minibatch_size': 3,
        'critic_lr': 0.05,
        'gamma': 0.99,  # Discount factor acts as effective horizon: 1/(1-gamma) gamma = 0.98 -> horizon ~= 50 min
        'use_gym_monitor': True,
        'render_env': True,
        'monitor_dir': monitor_dir,
        'buffer_size': 1000000,
        'summary_dir': current_summary_path,
        'max_episodes': 2,
        'max_episode_len': 60 * 24 / sensor_sample_time,
        'trained_models_path': (current_summary_path, 'test'),  # Format: (path, model extension e.g critic_test.joblib)
        # 'Load_models_path': None  # if None train new models, otherwise load models from path actor\critic.joblib
        'Load_models_path': (load_model, 'test')  # if None train new models, otherwise load models from path actor\critic.joblib
    }

    current_time = str(datetime.now())
    logging.info(f'Start Timestamp: {current_time}')
    args4print = json.dumps(args, sort_keys=True, indent=4)
    logging.info(f'Arguments:\n {args4print}')
    if mode == 'train':
        train_ddpg(args)
    if mode == 'inference':
        our_controller = DDPG_Controller(args['Load_models_path'])
        # init simulation
        our_controller.react
        animate = True
        patient_names = get_patients([1, 2])
        pump_name = get_insulin_pump(selection=2)
        # pump = InsulinPump.withName(pump_name)
        cgm_seed = 5
        cgm_sensor_name = get_cgm_sensor(selection=1)
        envs = our_build_envs(scenario, start_time, patient_names, cgm_sensor_name, cgm_seed, pump_name)
        env = envs[0]

        sim_time = 24
        controller = BBController()
        start_time = '0'
        sim = SimObj(envs, controller, sim_time, animate=True, path=None)

    logging.info(f'End Time: {str(datetime.now())}')
