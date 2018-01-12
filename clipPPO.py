import numpy as np
import tensorflow as tf
import tflearn as tfl
import gym
from collections import deque
from tensorflow.contrib.distributions import MultivariateNormalDiag


# Class that generates minibatches
class BatchGenerator:
    def __init__(self, data, minibatch_size):
        self.data = data
        self.minibatch_size = minibatch_size

    def iterate_once(self):
        per = np.random.permutation(np.arange(len(self.data[0])))
        if len(self.data[0]) % self.minibatch_size != 0:
            np.append(per, np.random.random_integers(0, len(self.data[0]), len(self.data[0]) % self.minibatch_size))
        for i in range(0, len(per), self.minibatch_size):
            p = per[i:i + self.minibatch_size]
            r = [None] * len(self.data)
            for j in range(len(self.data)):
                r[j] = self.data[j][p]
            yield tuple(r)


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
        self.s[self.i] = s
        self.a[self.i] = a
        self.r[self.i] = r
        self.t[self.i] = t
        self.s1[self.i] = s1
        self.i += 1

    def reset(self):
        self.i = 0

    def is_full(self):
        return self.i == self.capacity


# Logger that logs whats going on while training
class Logger:
    def __init__(self):
        self.reward_buffer = deque(maxlen=100)
        self.reward_buffer.append(0.)
        self.episode = 0
        self.avg_rew = float('-inf')
        self.best_avg_rew = self.avg_rew

    def update(self, b):
        for r, t in zip(b.r, b.t):
            self.reward_buffer[-1] += r
            if t:
                if len(self.reward_buffer) == self.reward_buffer.maxlen:
                    self.avg_rew = np.mean(self.reward_buffer)
                    if self.avg_rew > self.best_avg_rew:
                        self.best_avg_rew = self.avg_rew
                self.reward_buffer.append(0.)
                self.episode += 1

    def log(self):
        print('Episode %d, AccReward %.2f' % (self.episode, self.avg_rew))


# Class that creates and holds both networks (value network and policy network)
class Network(object):

    def __init__(self, state_dim, act_dim, hid_top):
        self.variables_v = tf.trainable_variables()
        self.input = tfl.input_data([None, state_dim])

        net = self.input
        for h in hid_top:
            net = tfl.fully_connected(net, h, activation='tanh')
        net = tfl.fully_connected(net, 1, activation='linear')
        self.vpred = tf.squeeze(net, axis=[1])
        self.variables_v = tf.trainable_variables()[len(self.variables_v):]

        self.variables_p = tf.trainable_variables()
        net = self.input
        for h in hid_top:
            net = tfl.fully_connected(net, h, activation='tanh')
        mean = tfl.fully_connected(net, act_dim, activation='linear')
        logstd = tf.Variable(initial_value=np.zeros(act_dim).astype(np.float32))
        self.variables_p = tf.trainable_variables()[len(self.variables_p):]

        self.mvn = MultivariateNormalDiag(mean, tf.exp(logstd))
        self.sample = self.mvn.sample()


# function that runs the environment and returns a buffer with all gathered data points
def traj_segment_generator(sess, net, env, size):
    b = Buffer(size, env.observation_space.shape[0], env.action_space.shape[0])
    s = env.reset()
    while True:
        a = sess.run(net.sample, {net.input: s[np.newaxis]})[0]
        s1, r, t, _ = env.step(a)
        b.add(s, a, r, t, s1)
        s = env.reset() if t else s1
        if b.is_full():
            yield b
            b.reset()


# function that calculates the standardized generalized advantage estimation
# as well as the estimated correct values for the value network
def calc_adv_and_vtarget(b, gamma, lam, net, sess):
    r = b.r
    t = b.t
    v = sess.run(net.vpred, {net.input: b.s})
    v1 = sess.run(net.vpred, {net.input: b.s1})
    T = b.capacity
    adv = np.empty(T, 'float32')
    lastadv = 0
    delta = r + gamma * v1 * (1 - t) - v
    for i in reversed(range(T)):
        adv[i] = lastadv = delta[i] + gamma * lam * (1 - t[i]) * lastadv
    vtarget = adv + v
    adv = (adv - adv.mean()) / adv.std()
    return adv, vtarget


# function that runs the clipPPO algorithm for 1500 episodes
# and returns the best average episode reward
def learn(env, sess, horizon, epsilon, ep_p, ep_v, lr_p, lr_v, mbs_p, mbs_v, gamma, lam):
    state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    # define all operators
    net = Network(state_dim, act_dim, [64, 64])
    neto = Network(state_dim, act_dim, [64, 64])

    adv_ph = tf.placeholder(dtype=tf.float32, shape=[None])
    vpred_ph = tf.placeholder(dtype=tf.float32, shape=[None])
    ac_ph = tf.placeholder(tf.float32, [None, act_dim])

    r = net.mvn.prob(ac_ph) / neto.mvn.prob(ac_ph)
    surr1 = r * adv_ph
    surr2 = tf.clip_by_value(r, 1.0 - epsilon, 1.0 + epsilon) * adv_ph
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
    vf_loss = tf.reduce_mean(tf.square(net.vpred - vpred_ph))

    opt_p = tf.train.AdamOptimizer(lr_p).minimize(pol_surr, var_list=net.variables_p)
    opt_v = tf.train.AdamOptimizer(lr_v).minimize(vf_loss, var_list=net.variables_v)
    copy_net_op = [o.assign(n) for n, o in zip(net.variables_p, neto.variables_p)]

    sess.run(tf.global_variables_initializer())

    logger = Logger()

    for b in traj_segment_generator(sess, net, env, horizon):

        # calc advantage and target values
        adv, vtarget = calc_adv_and_vtarget(b, gamma, lam, net, sess)

        # copy policy network
        sess.run(copy_net_op)

        # train both networks
        bg = BatchGenerator((b.s, b.a, adv), mbs_p)
        for _ in range(ep_p):
            for ms, ma, madv in bg.iterate_once():
                sess.run(opt_p, {net.input: ms, neto.input: ms, ac_ph: ma, adv_ph: madv})

        bg = BatchGenerator((b.s, vtarget), mbs_v)
        for _ in range(ep_v):
            for ms, mvpred in bg.iterate_once():
                sess.run(opt_v, {net.input: ms, vpred_ph: mvpred})

        # log everything
        logger.update(b)
        logger.log()
        if logger.episode >= 1500:
            break
    env.close()
    return logger.best_avg_rew

if __name__ == '__main__':
    tf.set_random_seed(1)
    np.random.seed(1)
    env = gym.make('Swimmer-v1')
    env.seed(1)
    env = gym.wrappers.Monitor(env, '/home/polarbart/Desktop/Swimmer-v1-t2/PPO_clip-9999-1', lambda x: x == 1500)
    print(learn(env, tf.Session(), 2048, .2, 10, 10, 3e-4, 3e-4, 64, 64, 0.9999, .95))
