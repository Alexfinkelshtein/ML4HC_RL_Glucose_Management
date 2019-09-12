from DDPG.ddpg import train_ddpg, DDPG_Controller
from gym.envs.registration import register
from simglucose.simulation.sim_engine import sim
from simglucose.analysis.report import report
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
from tqdm import tqdm

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


def random_meals(num_meals=3, time_min=0, time_max=10, size_min=5, size_max=70):
    times = []
    meals = []
    for i in range(num_meals):
        (time, size) = random_meal(time_min=time_min, time_max=time_max, size_min=size_min, size_max=size_max)
        if time not in times:  # avoid having two meals at same time
            meals.append((time, size))
            times.append(time)
    return meals


if __name__ == "__main__":
    # %% CONFIG\PATHS\LOGGER
    paths = yaml.load(open(r'PATHS.YAML'))
    base_path = paths['base_path']
    summaries_base = P.join(base_path, 'Results', 'Summaries')
    monitor_dir = P.join(base_path, 'Results', 'Monitor')
    # mode = 'train'
    # mode = 'inference'
    mode = "inference" if input("[1] for inference \n[2] for training\n") == '1' else "train"
    model_num = input("\n->Enter Model Number \n (0 if new model)")
    load_path = None if model_num == '0' else P.join(summaries_base, model_num)
    # load_path = P.join(summaries_base, '23')
    if load_path == None:
        current_summary_path = summary_path(summaries_base)
    else:
        current_summary_path = load_path
    logging.basicConfig(filename=os.path.join(current_summary_path, 'log.log'), level=logging.INFO)
    animate = False  # control simglucose graphic animations
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
    # scenario = CustomScenario(start_time=start_time, scenario=random_meals(num_meals=3))
    # %% REGISTER GYM ENVIRONMENT

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
        # 'Load_models_path': None,  # if None train new models, otherwise load models from path actor\critic.joblib
        'Load_models_path': (load_path, 'test'),
        # if None train new models, otherwise load models from path actor\critic.joblib
    }
    current_time = str(dt.now())
    logging.info(f'Start Timestamp: {current_time}')
    args4print = json.dumps(args, sort_keys=True, indent=4)
    logging.info(f'Arguments:\n {args4print}')

    # acceptable parameters at simglucose_gym_env
    register(id='simglucose-adolescent2-v0',
             entry_point='simglucose.envs:T1DSimEnv',
             kwargs={'patient_name': patient_name, 'custom_scenario': scenario, 'animate': animate,
                     'controller_name': controller_name, 'results_path': P.join(current_summary_path, 'Analysis_DDPG')
                     }
             )

    gym_env = gym.make('simglucose-adolescent2-v0')

    if mode == 'train':
        if int(input("Load Buffer Memory?\n[0]No\n[1]Yes\n")):
            args['buffer_path'] = P.join(base_path, 'Results', 'Buffer_memory')
        else:
            args['buffer_path'] = None
        single_scenario = True if input(
            "Choose training type:\n[0] Single Scenario\n[1] Multiple Scenarios\n") == '0' else False
        if single_scenario:
            print("[INFO] Training on Single Scenario")
            train_ddpg(args)
        else:
            n_scenarios = 20
            print(f"[INFO] Training on {n_scenarios} Scenarios")
            for i in tqdm(range(1, n_scenarios + 1)):
                id = f'simglucose-adolescent2-v{i}'
                register(id=id,
                         entry_point='simglucose.envs:T1DSimEnv',
                         kwargs={'patient_name': patient_name,
                                 'custom_scenario': CustomScenario(start_time=start_time,
                                                                   scenario=random_meals(
                                                                       num_meals=3, time_max=5)),
                                 'animate': animate,
                                 'controller_name': controller_name,
                                 'results_path': P.join(current_summary_path, 'Analysis_DDPG')
                                 }
                         )
                gym_env = gym.make(id)
                args['env'] = id
                train_ddpg(args)
                args['Load_models_path'] = (current_summary_path, 'test')
    if mode == 'inference':
        with tf.Session() as sess:
            analysis_path = P.join(current_summary_path, 'Analysis')
            # init simulation
            patient_names = get_patients([patient_number])
            pump_name = get_insulin_pump(selection=2)
            cgm_seed = 5
            cgm_sensor_name = get_cgm_sensor(selection=1)
            sim_time = datetime.timedelta(hours=8)  # datetime.time(23)
            control = input("Choose controller:\n[1]DDPG\n[2]BB\n")
            if control == '2':
                # Original Controller
                controller = BBController()
                envs = our_build_envs(scenario, start_time, patient_names, cgm_sensor_name, cgm_seed, pump_name,
                                      controller_name='BB Controller', results_path=analysis_path + '_BB')
                env_BB = envs[0]
                sim1 = SimObj(env_BB, controller, sim_time, animate=animate, path=analysis_path + '_BB')
                results1 = sim(sim1)
                report(results1, analysis_path + '_BB')
                env_BB.render()
                env_BB.viewer.close()

            # Our Controller
            else:
                state_dim = gym_env.observation_space.shape[0]
                action_dim = gym_env.action_space.shape[0]
                action_bound = gym_env.action_space.high
                actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                                     float(args['actor_lr']), float(args['tau']),
                                     int(args['minibatch_size']))
                sess.run(tf.global_variables_initializer())
                actor.restore(sess, P.join(args['Load_models_path'][0], f"actor_{args['Load_models_path'][1]}"))
                our_controller = DDPG_Controller(actor=actor)
                sim2 = SimObj(gym_env, our_controller, sim_time, animate=animate, path=analysis_path + '_DDPG')
                results2 = sim(sim2)
                report(results2, analysis_path + '_DDPG')
                gym_env.env.render()
                gym_env.env.viewer.close()
    logging.info(f'End Time: {str(dt.now())}')
