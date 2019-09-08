import logging
import time
import os
from DDPG.ddpg import DDPG_Controller

pathos = True
try:
    from pathos.multiprocessing import ProcessPool as Pool
except ImportError:
    print('You could install pathos to enable parallel simulation.')
    pathos = False

logger = logging.getLogger(__name__)


class SimObj(object):
    def __init__(self,
                 env,
                 controller,
                 sim_time,
                 animate=True,
                 path=None):
        self.env = env
        self.controller = controller
        self.sim_time = sim_time
        self.animate = animate
        self._ctrller_kwargs = None
        self.path = path

    def simulate(self):
        if isinstance(self.controller, DDPG_Controller):
            obs = self.env.reset()
            end_time = self.env.env.scenario.start_time + self.sim_time
            current_time = self.env.env.scenario.start_time
        else:
            obs, reward, done, info = self.env.reset()
            end_time = self.env.scenario.start_time + self.sim_time
            current_time = self.env.scenario.start_time
        tic = time.time()

        while current_time < end_time:
            if self.animate:
                if isinstance(self.controller, DDPG_Controller):
                    self.env.env.render()
                else:
                    self.env.render()
            if isinstance(self.controller, DDPG_Controller):
                action = self.controller.policy(obs)
            else:
                action = self.controller.policy(obs, reward, done, **info)

            obs, reward, done, info = self.env.step(action)
            if isinstance(self.controller, DDPG_Controller):
                current_time = self.env.env.time
            else:
                current_time = self.env.time
        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))

    def results(self):
        try:
            return self.env.show_history()
        except:
            return self.env.env.show_history()

    def save_results(self):
        df = self.results()
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        try:
            filename = os.path.join(self.path, str(self.env.patient.name) + '.csv')
        except:
            filename = os.path.join(self.path, str(self.env.env.patient.name) + '.csv')
        df.to_csv(filename)

    def reset(self):
        self.env.reset()
        self.controller.reset()


def sim(sim_object):
    print("Process ID: {}".format(os.getpid()))
    print('Simulation starts ...')
    sim_object.simulate()
    sim_object.save_results()
    print('Simulation Completed!')
    return sim_object.results()


def batch_sim(sim_instances, parallel=False):
    tic = time.time()
    if parallel and pathos:
        with Pool() as p:
            results = p.map(sim, sim_instances)
    else:
        if parallel and not pathos:
            print('Simulation is using single process even though parallel=True.')
        results = [sim(s) for s in sim_instances]
    toc = time.time()
    print('Simulation took {} sec.'.format(toc - tic))
    return results
