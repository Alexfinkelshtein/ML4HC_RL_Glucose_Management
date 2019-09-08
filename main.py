from DDPG.ddpg import train_ddpg, DDPG_Controller
from gym.envs.registration import register
import os
import numpy as np
# imports
from simglucose.simulation.user_interface import *
import gym
import logging
import json
from pandas.plotting import register_matplotlib_converters
import tensorflow as tf
from DDPG.ddpg import ActorNetwork
import os.path as P

register_matplotlib_converters()
from datetime import datetime as dt
import datetime
import yaml
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


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
    paths = yaml.load(open(r'PATHS.YAML'))
    base_path = paths['base_path']
    summaries_base = P.join(base_path, 'Results', 'Summaries')
    monitor_dir = P.join(base_path, 'Results', 'Monitor')
    # mode = 'train'
    # mode = 'inference'
    mode = "inference" if input("[1] for inference \n [2] for training") == '1' else "train"
    model_num = input("\n->Enter Model Number \n (0 if new model)")
    load_path = None if model_num == '0' else P.join(summaries_base, model_num)
    # load_path = P.join(summaries_base, '23')
    if load_path == None:
        current_summary_path = summary_path(summaries_base)
    else:
        current_summary_path = load_path
    logging.basicConfig(filename=os.path.join(current_summary_path, 'log.log'), level=logging.INFO)
    animate = True  # control simglucose graphic animations
    patient_number = 2
    patient_name = f'adolescent#00{patient_number}'
    controller_name = 'DDPG Controller'



    # %% create scenario
    now = dt.now()
    start_hour = timedelta(hours=float(0))
    start_time = dt.combine(now.date(), dt.min.time()) + start_hour
    # meals_ = random_meals(num_meals=4)  # example for randomized scenario creation
    meals = [(1, 60), (3, 80), (5, 60)]  # format: list of tuples, where (meal_time, meal_size [grams])
    scenario = CustomScenario(start_time=start_time, scenario=meals)
    # %% REGISTER GYM ENVIRONMENT

    # acceptable parameters at simglucose_gym_env
    register(id='simglucose-adolescent2-v0',
             entry_point='simglucose.envs:T1DSimEnv',
             kwargs={'patient_name': patient_name, 'custom_scenario': scenario, 'animate': animate,
                     'controller_name': controller_name}
             )
    gym_env = gym.make('simglucose-adolescent2-v0')

    # %% DDPG Controller
    sensor_sample_time = 3

    args = {
        'env': f'simglucose-adolescent{patient_number}-v0',
        'random_seed': 1234,
        'actor_lr': 0.0005,
        'tau': 0.01,
        'minibatch_size': 64,
        'critic_lr': 0.0025,
        'gamma': 0.99,  # Discount factor acts as effective horizon: 1/(1-gamma) gamma = 0.98 -> horizon ~= 50 min
        'use_gym_monitor': True,
        'render_env': True,
        'monitor_dir': monitor_dir,
        'buffer_size': 1000000,
        'summary_dir': current_summary_path,
        'max_episodes': 2000,
        'max_episode_len': 60 * 24 / sensor_sample_time,
        'trained_models_path': (current_summary_path, 'test'),  # Format: (path, model extension e.g critic_test.joblib)
        'Load_models_path': None,  # if None train new models, otherwise load models from path actor\critic.joblib
        'Load_models_path': (load_path, 'test')
        # if None train new models, otherwise load models from path actor\critic.joblib
    }

    current_time = str(dt.now())
    logging.info(f'Start Timestamp: {current_time}')
    args4print = json.dumps(args, sort_keys=True, indent=4)
    logging.info(f'Arguments:\n {args4print}')
    if mode == 'train':
        train_ddpg(args)
    if mode == 'inference':
        with tf.Session() as sess:
            analysis_path = P.join(current_summary_path, 'Analysis')
            # init simulation
            patient_names = get_patients([patient_number])
            pump_name = get_insulin_pump(selection=2)
            cgm_seed = 5
            cgm_sensor_name = get_cgm_sensor(selection=1)
            sim_time = datetime.timedelta(hours=8)  # datetime.time(23)
            # Original Controller
            controller = BBController()
            envs = our_build_envs(scenario, start_time, patient_names, cgm_sensor_name, cgm_seed, pump_name,
                                  controller_name='BB Controller')
            env1 = envs[0]
            sim1 = SimObj(env1, controller, sim_time, animate=animate, path=analysis_path)
            from simglucose.simulation.sim_engine import sim
            from simglucose.analysis.report import report

            # simulate(sim_time=sim_time, scenario=scenario, controller=controller, start_time=start_time,
            #          save_path=current_summary_path, animate=animate,
            #          parallel=True,
            #          controller_name='Their Controller',
            #          envs=envs)
            results1 = sim(sim1)
            # report(results1, analysis_path)
            # Our Controller
            state_dim = gym_env.observation_space.shape[0]
            action_dim = gym_env.action_space.shape[0]
            action_bound = gym_env.action_space.high
            actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                                 float(args['actor_lr']), float(args['tau']),
                                 int(args['minibatch_size']))
            sess.run(tf.global_variables_initializer())
            actor.restore(sess, P.join(args['Load_models_path'][0], f"actor_{args['Load_models_path'][1]}"))
            our_controller = DDPG_Controller(actor=actor)
            sim2 = SimObj(gym_env, our_controller, sim_time, animate=animate, path=analysis_path)
            results2 = sim(sim2)
            # report(results2, analysis_path)
    logging.info(f'End Time: {str(dt.now())}')
