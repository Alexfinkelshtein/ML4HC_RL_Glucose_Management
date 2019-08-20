"""
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp
import matplotlib.pyplot as plt
import io
from PIL import Image

from DDPG.replay_buffer import ReplayBuffer
from tqdm import tqdm


# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 1200, activation='relu')
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 900, activation='relu')
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-1, maxval=1)  # TODO consider xavier initialization
        # w_init = tflearn.initializations.uniform(minval=0, maxval=0.03)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='relu', weights_init=w_init)

        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1], name='predicted_q')

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 1200, activation='relu')
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 900)
        t2 = tflearn.fully_connected(action, 900)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-3e-3, maxval=3e-3)
        out = tflearn.fully_connected(net, 1, weights_init=w_init, name='out')
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        ans = self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })
        return ans

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
    # def __init__(self, mu, sigma=5, theta=.15, dt=1e-2, x0=None):  # CHANGED
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)

    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)

    loss_critic = tf.Variable(0.)
    tf.summary.scalar("loss_critic", loss_critic)

    image = tf.Variable(np.zeros((1, 480, 640, 4)), expected_shape=(None, 480, 640, 4), dtype=tf.uint8)
    tf.summary.image("CGM plot", image)

    summary_vars = [episode_reward, episode_ave_max_q, loss_critic, image]
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    #  OUR ASSUMPTIONS
    ins_bound_param = 15

    for i in tqdm(range(int(args['max_episodes']))):
        T1DSimEnv = env.env.env
        s = env.reset()
        meal_times = [int(meal[0] * 60.0 / 3) for meal in T1DSimEnv.scenario.scenario]
        meal_sizes = [meal[1] for meal in T1DSimEnv.scenario.scenario]
        # s = env.reset().observation  # CHANGED

        ep_reward = 0
        ep_rewards = []
        ep_cgm = []
        ep_ins = []
        ep_ave_max_q = 0
        loss_critic = 0
        loss_critic_ep = []
        a_basal_hist = []
        a_bolus_hist = []
        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render(mode='human')

            # Added exploration noise
            # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()  # makes sure noise doesnt cross bound
            # if i % 2 == 0:
            #     a = [np.array([30])]
            # else:
            #     a = [np.array([0])]
            #     a = [np.array([-15, -15])]
            #             a = [env.action_space.sample()]
            s2, r, terminal, info = env.step(a[0])

            # window_size = int(60 / T1DSimEnv.sample_time)  # Horizon
            # BG_last_hour = T1DSimEnv.CGM_hist[-window_size:]  # Blood Glucose Last Hour
            # IN_last_hour = T1DSimEnv.insulin_hist[-window_size:]  # Insulin Last Hour
            # if j < window_size:  # Padding
            #     pad_size_BG = window_size - len(BG_last_hour)
            #     pad_size_IN = window_size - len(IN_last_hour)
            #     BG_last_hour = [BG_last_hour[0]] * pad_size_BG + BG_last_hour  # Blood Glucose Last Hour
            #     IN_last_hour = [IN_last_hour[0]] * pad_size_IN + IN_last_hour  # Insulin Last Hour

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))
                ep_ave_max_q += np.amax(predicted_q_value)
                loss_critic += np.mean(np.square(predicted_q_value - y_i))

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])
                # Update target networks

                actor.update_target_network()
                critic.update_target_network()
            # basal_ins = max(min(a[0][0], ins_bound_param), -1 * ins_bound_param)
            # basal_ins = a[0][0]
            # bolus_ins = max(min(a[0][1], ins_bound_param), -1 * ins_bound_param)
            # insulin = basal_ins + bolus_ins + ins_bound_param * 2
            # insulin = basal_ins + ins_bound_param
            s = s2
            ep_reward += r
            ep_rewards += [r]
            ep_cgm += [s]
            loss_critic_ep += [loss_critic]
            # ep_ins += [insulin]
            # if insulin < 0:
            #     print(insulin)
            #     print(f"negative at iteration: {j}")
            # a_basal_hist += [basal_ins]  # TODO make sure basal
            # a_bolus_hist += [bolus_ins]  # TODO make sure bolus

            if terminal:
                plt.figure()
                ax = plt.gca()
                plt.plot(T1DSimEnv.CGM_hist, label="CGM plot")
                plt.plot(T1DSimEnv.BG_hist, label="BG plot")
                plt.plot(T1DSimEnv.CHO_hist, label="CHO plot")
                plt.plot(T1DSimEnv.insulin_hist, label="Insulin plot")
                plt.plot(ep_rewards, label="Reward plot")
                plt.plot(np.array(range(len(T1DSimEnv.CHO_hist)))[meal_times], np.array(T1DSimEnv.CHO_hist)[meal_times],
                         'g*', label="Meal")
                # TODO: add annotations with meal sizes
                plt.title("Episode CGM and Insulin vs Time")
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)

                image = Image.open(buf)
                image = np.array(image, dtype='uint8')
                image = np.expand_dims(image, 0)

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j),
                    summary_vars[2]: float(np.mean(loss_critic_ep)),
                    summary_vars[3]: image,

                })

                writer.add_summary(summary_str, i)
                writer.flush()
                plt.close()
                print('| Reward: {:.4f} | Episode: {:d} | Qmax: {:.4f}'.format(float(ep_reward), i,
                                                                               (ep_ave_max_q / float(j))))
                break


def main(args):
    with tf.Session() as sess:

        env = gym.make(args['env'])
        # env = args['env']
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        # TODO: decide about action space with respect to symmetry
        # TODO consider:
        # def rescale_actions(tanh_output, low, high):
        #     range = high - low
        #     return tanh_output * range / 2 + (low + (0.5 * range))
        # assert (env.action_space.high == -env.action_space.low)  # TODO tackle symmetry

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, args, actor, critic, actor_noise)

        if args['use_gym_monitor']:
            env.monitor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)
