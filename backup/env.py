from simglucose.patient.t1dpatient import Action
from simglucose.analysis.risk import risk_index, risk_index2
import pandas as pd
from datetime import timedelta
import logging
from collections import namedtuple
from simglucose.simulation.rendering import Viewer
import numpy as np
from gym import spaces

try:
    from rllab.envs.base import Step
except ImportError:
    _Step = namedtuple("Step", ["observation", "reward", "done", "info"])


    def Step(observation, reward, done, **kwargs):
        """
        Convenience method creating a namedtuple with the results of the
        environment.step method.
        Put extra diagnostic info in the kwargs
        """
        return _Step(observation, reward, done, kwargs)

Observation = namedtuple('Observation', ['CGM', 'INSULIN', 'CHO'])
logger = logging.getLogger(__name__)


def risk_diff(BG_last_hour):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index2([BG_last_hour[-1]], 1)  # Horizon
        _, _, risk_prev = risk_index2([BG_last_hour[-2]], 1)  # Horizon
        normalized = lambda x: x / 120
        # normalized = lambda x: x
        # return normalized(risk_prev - risk_current)
        return -normalized(risk_current)


normalize_cgm = lambda x: 2 * (x - 39) / (600 - 39) - 1
normalize_ins = lambda x: x / 30
normalize_cho = lambda x: x / 1000


class T1DSimEnv(object):

    def __init__(self, patient, sensor, pump, scenario):
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        # self.observation_space = spaces.Box(20, 350, (1, 1))
        # self.action_space = spaces.Box(0, 400, (2, 1))
        self._reset()

    # def seed(self, rand_seed):
    #     self.sensor.seed = rand_seed

    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action):
        # current action
        action_offset = 15  # ASSUMPTION: max pump 30
        patient_action = self.scenario.get_action(self.time)
        basal = self.pump.basal(action.basal[0])  # making sure not out fo bounds
        # basal = self.pump.basal(max(min(action.basal, action_offset), -action_offset))  # making sure not out fo bounds
        # basal = self.pump.basal(action[0])  # CHANGED
        # bolus = self.pump.bolus(max(min(action.bolus, action_offset), -action_offset))  # making sure not out fo bounds
        # bolus = self.pump.bolus(action[1])  # CHANGED
        # insulin = basal + bolus + action_offset * 2  # CHANGED
        insulin = basal
        CHO = patient_action.meal
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)

        # State update
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)

        return CHO, insulin, BG, CGM

    def step(self, action, reward_fun=risk_diff):
        '''
        action is a namedtuple with keys: basal, bolus
        '''
        CHO = 0.0
        insulin = 0.0
        BG = 0.0
        CGM = 0.0

        CHO_norm = 0.0
        insulin_norm = 0.0
        CGM_norm = 0.0

        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action)
            if tmp_insulin < 0:
                print(tmp_insulin)
                print(self.patient.t)
            CHO += tmp_CHO / self.sample_time
            insulin += tmp_insulin / self.sample_time
            BG += tmp_BG / self.sample_time
            CGM += tmp_CGM / self.sample_time
            # CHO_norm += normalize_cho(tmp_CHO / self.sample_time)
            # CGM_norm += normalize_cgm(tmp_CGM / self.sample_time)
            # insulin_norm += normalize_ins(tmp_insulin / self.sample_time)

        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)

        # Record current action
        self.CHO_hist.append(CHO)
        # self.CHO_hist.append(CHO_norm)
        self.insulin_hist.append(insulin)
        # self.insulin_hist.append(insulin_norm)

        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        # self.CGM_hist.append(CGM_norm)
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)

        # Compute reward, and decide whether game is over
        window_size = int(60 / self.sample_time)  # Horizon
        BG_last_hour = self.CGM_hist[-window_size:]
        reward = reward_fun(BG_last_hour)
        # done = BG < 70 or BG > 350
        done = self.patient.t == (24 * 60) / self.sample_time - 1 * self.sample_time
        cgm_s = self.CGM_hist[-window_size:]
        ins_s = self.insulin_hist[-window_size:]
        cho_s = self.CHO_hist[-window_size:]
        if min(len(self.CGM_hist), len(self.insulin_hist)) < window_size:  # Padding
            pad_size_cgm = max(window_size - len(self.CGM_hist), 0)
            pad_size_IN = max(window_size - len(self.insulin_hist), 0)
            pad_size_CHO = max(window_size - len(self.CHO_hist), 0)
            cgm_s = [self.CGM_hist[0]] * pad_size_cgm + self.CGM_hist  # Blood Glucose Last Hour
            ins_s = [self.insulin_hist[0]] * pad_size_IN + self.insulin_hist  # Insulin Last Hour
            cho_s = [self.CHO_hist[0]] * pad_size_CHO + self.CHO_hist  # Insulin Last Hour

        obs = Observation(CGM=cgm_s, INSULIN=ins_s, CHO=cho_s)

        return Step(
            observation=obs,
            reward=reward,
            done=done,
            sample_time=float(self.sample_time),
            patient_name=self.patient.name,
            meal=CHO,
            patient_state=self.patient.state)

    def _reset(self):
        self.sample_time = self.sensor.sample_time
        self.viewer = None

        BG = self.patient.observation.Gsub
        horizon = 1  # TODO understand
        LBGI, HBGI, risk = risk_index([BG], horizon)
        CGM = self.sensor.measure(self.patient)
        self.time_hist = [self.scenario.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM]
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.CHO_hist = []
        self.insulin_hist = []
        self.CHO_hist = []
        self.insulin_hist = []

    def reset(self):
        self.patient.reset()
        self.sensor.reset()
        self.pump.reset()
        self.scenario.reset()
        self._reset()
        CGM = self.sensor.measure(self.patient)
        obs = Observation(CGM=[normalize_cgm(CGM)] * 20, INSULIN=[0] * 20, CHO=[0] * 20)
        return Step(
            observation=obs,
            reward=0,
            done=False,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=0,
            patient_state=self.patient.state)

    def render(self, close=True):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario.start_time, self.patient.name, )

        self.viewer.render(self.show_history())

    def show_history(self):
        df = pd.DataFrame()
        df['Time'] = pd.Series(self.time_hist)
        df['BG'] = pd.Series(self.BG_hist)
        df['CGM'] = pd.Series(self.CGM_hist)
        df['CHO'] = pd.Series(self.CHO_hist)
        df['insulin'] = pd.Series(self.insulin_hist)
        df['LBGI'] = pd.Series(self.LBGI_hist)
        df['HBGI'] = pd.Series(self.HBGI_hist)
        df['Risk'] = pd.Series(self.risk_hist)
        df = df.set_index('Time')
        return df
