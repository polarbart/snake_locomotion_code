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
        self.episode = self.beta = self.kl = self.var = 0
        self.avg_rew = float('-inf')
        self.best_avg_rew = self.avg_rew

    def update(self, b, beta, kl, var):
        for r, t in zip(b.r, b.t):
            self.reward_buffer[-1] += r
            if t:
                if len(self.reward_buffer) == self.reward_buffer.maxlen:
                    self.avg_rew = np.mean(self.reward_buffer)
                    if self.avg_rew > self.best_avg_rew:
                        self.best_avg_rew = self.avg_rew
                self.reward_buffer.append(0.)
                self.episode += 1
        self.beta = beta
        self.kl = kl
        self.var = var

    def log(self):
        print('Episode %d, AccReward %.2f, beta %.6f, kl %.6f, var %s'
              % (self.episode, np.mean(self.reward_buffer), self.beta, self.kl, str(self.var)))


# Class that creates and holds both networks (value network and policy network)
class Network(object):
    def __init__(self, state_dim, act_dim):
        self.input = tfl.input_data([None, state_dim])

        self.v_variables = tf.trainable_variables()
        net = self.input
        for h in [64, 64]:  # 80, 40, 5
            net = tfl.fully_connected(net, h, activation='tanh')
        net = tfl.fully_connected(net, 1, activation='linear')
        self.vpred = tf.squeeze(net, axis=[1])
        self.v_variables = tf.trainable_variables()[len(self.v_variables):]

        self.p_variables = tf.trainable_variables()
        net = self.input
        for h in [64, 64]:  # 80, 50, 20
            net = tfl.fully_connected(net, h, activation='tanh')
        self.mean = tfl.fully_connected(net, act_dim, activation='linear')
        self.var = tf.exp(tf.Variable(initial_value=np.zeros(act_dim).astype(np.float32)))

        self.mvn = MultivariateNormalDiag(self.mean, self.var)
        self.sample = self.mvn.sample()
        self.p_variables = tf.trainable_variables()[len(self.p_variables):]


# Returns operator that calcs the KL divergence between
# two multivariate Gaussians with diagonal covariance matrix
def kl_diag_mvn(mu0, var0, mu1, var1):
    d = mu0.shape.as_list()[1]
    return .5 * (tf.reduce_sum(var0 / var1) + tf.reduce_sum(tf.square(mu1 - mu0) / var1, 1) - d
                 + tf.log(tf.reduce_prod(var1 / var0)))


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


# function that runs the KLPPO algorithm for 1500 episodes
# and returns the best average episode reward
def learn(env, sess, horizon, target_kl, ep_p, ep_v, lr_p, lr_v, mbs_p, mbs_v, gamma, lam, alpha, beta_low,
          beta_high):

    # define all operators
    state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    net = Network(state_dim, act_dim)
    neto = Network(state_dim, act_dim)

    adv_ph = tf.placeholder(dtype=tf.float32, shape=[None])
    vtarget_ph = tf.placeholder(dtype=tf.float32, shape=[None])
    ac_ph = tf.placeholder(tf.float32, [None, act_dim])
    beta_ph = tf.placeholder(tf.float32, [])

    kl_op = tf.reduce_mean(kl_diag_mvn(neto.mean, neto.var, net.mean, net.var))
    r = net.mvn.prob(ac_ph) / neto.mvn.prob(ac_ph)
    l1 = -r * adv_ph
    l2 = beta_ph * kl_op
    pol_loss = tf.reduce_mean(l1 + l2)
    vf_loss = tf.reduce_mean(tf.square(net.vpred - vtarget_ph))

    opt_p = tf.train.AdamOptimizer(lr_p).minimize(pol_loss, var_list=net.p_variables)
    opt_v = tf.train.AdamOptimizer(lr_v).minimize(vf_loss, var_list=net.v_variables)
    copy_net_op = [o.assign(n) for n, o in zip(net.p_variables, neto.p_variables)]

    sess.run(tf.global_variables_initializer())

    logger = Logger()

    beta = 1.
    for b in traj_segment_generator(sess, net, env, horizon):

        # calc advantage and target values
        adv, vtarget = calc_adv_and_vtarget(b, gamma, lam, net, sess)

        # copy policy network
        sess.run(copy_net_op)

        # train policy and value network
        bg = BatchGenerator((b.s, b.a, adv), mbs_p)
        for i in range(ep_p):
            for ms, ma, madv in bg.iterate_once():
                sess.run(opt_p, {net.input: ms, neto.input: ms, ac_ph: ma, adv_ph: madv,
                                 beta_ph: beta})

        bg = BatchGenerator((b.s, vtarget), mbs_v)
        for i in range(ep_v):
            for ms, mvtarget in bg.iterate_once():
                sess.run(opt_v, {net.input: ms, vtarget_ph: mvtarget})

        # update beta
        kl = sess.run(kl_op, {net.input: ms, neto.input: ms})
        if kl > beta_high * target_kl:
            beta = min(35., beta * alpha)
        elif kl < beta_low * target_kl:
            beta = max(1. / 35., beta / alpha)

        logger.update(b, beta, kl, sess.run(net.var))
        logger.log()
        if logger.episode >= 1500:
            break
    env.close()
    return logger.best_avg_rew


def test(seed, save_dir, horizon, target_kl, ep_p, ep_v, lr_p, lr_v, mbs_p, mbs_v, lam, gamma=1.):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make('Swimmer-v1')
    env.seed(seed)
    env = gym.wrappers.Monitor(env, save_dir, lambda x: x == 1500)
    return learn(env, tf.Session(), horizon, target_kl, ep_p, ep_v, lr_p, lr_v, mbs_p, mbs_v, gamma, lam, 1.5, .5, 2.)

if __name__ == '__main__':
    print(test(0, '/tmp/Swimmer-v1/0-99/', 8192, 0.007, 4, 8, 5e-4, 5e-3, 2048, 256, .95, 0.99))
