import numpy as np
import tensorflow as tf
import tflearn as tfl
import gym
from collections import deque
from tensorflow.contrib.distributions import MultivariateNormalDiag
import time


# Buffer with fixed capacity to save all data points for training
class Buffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.s = np.empty([capacity, state_dim])
        self.a = np.empty([capacity, action_dim])
        self.r = np.empty([capacity])
        self.t = np.empty([capacity], dtype=bool)
        self.s1 = np.empty([capacity, state_dim])
        self.i = 0

    def add(self, s, a, r, t, s1):
        assert len(s) == len(a) == len(r) == len(t) == len(s1)
        l = len(s)
        self.s[self.i:self.i + l] = s
        self.a[self.i:self.i + l] = a
        self.r[self.i:self.i + l] = r
        self.t[self.i:self.i + l] = t
        self.s1[self.i:self.i + l] = s1
        self.i += l

    def reset(self):
        self.i = 0

    def is_full(self):
        return self.i == self.capacity


# Logger that logs whats going on while training
class Logger:
    def __init__(self, num_envs):
        self.reward_buffer = deque(maxlen=100)
        self.reward_buffer.append(0.)
        self.best_100_avg = float('-inf')
        self.best_100_avg_list = []
        self.episode = 0
        self.rdy_to_log = False
        self.num_envs = num_envs
        self.time = time.time()

    def update(self, b):
        for r, t in zip(b.r[0:-1:self.num_envs], b.t[0:-1:self.num_envs]):
            self.reward_buffer[-1] += r
            if t:
                if len(self.reward_buffer) == self.reward_buffer.maxlen:
                    self.best_100_avg = max(self.best_100_avg, np.mean(self.reward_buffer))
                self.best_100_avg_list.append(self.best_100_avg)
                self.reward_buffer.append(0.)
                self.episode += 1
                self.rdy_to_log = True

    def log(self):
        if self.rdy_to_log:
            print('Episode %d, AccReward %.2f, Time %-2f' % (self.episode, np.mean(self.reward_buffer), time.time()-self.time))
            self.rdy_to_log = False
            self.time = time.time()


# Class that creates and holds both networks (value network and policy network)
class Network(object):
    def __init__(self, state_dim, act_dim):
        self.input = tfl.input_data([None, state_dim])

        self.variables_v = tf.trainable_variables()
        net = self.input
        for h in [64, 64]:
            net = tfl.fully_connected(net, h, activation='tanh')
        net = tfl.fully_connected(net, 1, activation='linear')
        self.vpred = tf.squeeze(net, axis=[1])
        self.variables_v = tf.trainable_variables()[len(self.variables_v):]

        self.variables_p = tf.trainable_variables()
        net = self.input
        for h in [64, 64]:
            net = tfl.fully_connected(net, h, activation='tanh')
        mean = tfl.fully_connected(net, act_dim, activation='linear')
        logstd = tf.Variable(initial_value=np.zeros(act_dim).astype(np.float32))
        self.variables_p = tf.trainable_variables()[len(self.variables_p):]

        self.mvn = MultivariateNormalDiag(mean, tf.exp(logstd))
        self.sample = self.mvn.sample()


# function that runs all environments and returns a buffer with all gathered data points
def traj_segment_generator(sess, net, envs, size):
    l = len(envs)
    b = Buffer(size * l, envs[0].observation_space.shape[0], envs[0].action_space.shape[0])
    s = [e.reset() for e in envs]
    while True:
        a = sess.run(net.sample, {net.input: s})
        s1, r, t = list(zip(*[envs[i].step(a[i])[0:3] for i in range(l)]))
        b.add(s, a, r, t, np.vstack(s1))
        s = [envs[i].reset() if t[i] else s1[i] for i in range(l)]
        if b.is_full():
            yield b
            b.reset()


# function that calculates the standardized generalized advantage estimation
# as well as the estimated correct values for the value network
def calc_adv_and_vtarget(b, l, gamma, lam, net, sess):
    r = b.r
    t = b.t
    v = sess.run(net.vpred, {net.input: b.s})
    v1 = sess.run(net.vpred, {net.input: b.s1})
    T = b.capacity
    adv = np.empty(T, 'float32')
    lastadv = [0.] * l
    delta = r + gamma * v1 * (1 - t) - v
    for i in reversed(range(T // l)):
        x = i * l
        for j in reversed(range(l)):
            adv[x + j] = lastadv[j] = delta[x + j] + gamma * lam * (1 - t[x + j]) * lastadv[j]
    vtarget = adv + v
    adv = (adv - adv.mean()) / adv.std()
    return adv, vtarget


# function that runs the A2C algorithm for 1500 episodes
# and returns the best average episode reward
def learn(envs, sess, horizon, lr_p, lr_v, gamma, lam):

    # define all operators
    state_dim, act_dim = envs[0].observation_space.shape[0], envs[0].action_space.shape[0]

    net = Network(state_dim, act_dim)

    adv_ph = tf.placeholder(dtype=tf.float32, shape=[None])
    vtarget_ph = tf.placeholder(dtype=tf.float32, shape=[None])
    ac_ph = tf.placeholder(tf.float32, [None, act_dim])

    loss_p = tf.reduce_mean(net.mvn.log_prob(ac_ph) * adv_ph)
    loss_v = tf.reduce_mean(tf.square(net.vpred - vtarget_ph))

    opt_p = tf.train.AdamOptimizer(lr_p).minimize(-loss_p, var_list=net.variables_p)
    opt_v = tf.train.AdamOptimizer(lr_v).minimize(loss_v, var_list=net.variables_v)

    sess.run(tf.global_variables_initializer())

    logger = Logger(len(envs))

    for b in traj_segment_generator(sess, net, envs, horizon):

        # calc advantage and target values
        adv, vtarget = calc_adv_and_vtarget(b, len(envs), gamma, lam, net, sess)

        # train both networks
        sess.run([opt_p, opt_v], {net.input: b.s, ac_ph: b.a, adv_ph: adv, vtarget_ph: vtarget})

        logger.update(b)
        logger.log()
        if logger.episode > 1500:
            break
    for e in envs:
        e.close()
    return logger.best_100_avg


def test_it(seed, save_dir, num_envs, horizon, lr_p, lr_v, lam, gamma=0.999):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env_name = 'Swimmer-v1'
    env = gym.make(env_name)
    env = gym.wrappers.Monitor(env, save_dir, lambda t: t == 1499)
    envs = [env] + [gym.make(env_name) for _ in range(num_envs - 1)]
    for e in envs:
        e.seed(seed)
    return learn(envs, tf.Session(), horizon, lr_p, lr_v, gamma, lam)

if __name__ == '__main__':
    print(test_it(0, '/tmp/Swimmer-v1-t2/A2C', 16, 5, 1e-4, 1e-4, 0.95, 0.99))
