import tensorflow as tf
import tflearn as tfl
import numpy as np
import random
import time
import gym


# Class that creates and holds all four networks (Q network and policy network)
class Network:
    def __init__(self, obs_dim, act_dim):
        self.aq_in, self.a_out, self.a_trainable_vars = \
            Network.create_a_net(obs_dim, act_dim)
        self.aqp_in, self.ap_out, self.ap_trainable_vars = \
            Network.create_a_net(obs_dim, act_dim)

        _, _, self.q_out, self.q_trainable_vars = \
            Network.create_q_net(obs_dim, act_dim, self.aq_in, self.a_out)
        _, _, self.qp_out, self.qp_trainable_vars = \
            Network.create_q_net(obs_dim, act_dim, self.aqp_in, self.ap_out)

    @staticmethod
    def create_a_net(obs_dim, act_dim):
        hidden_layer_top = [64, 64]
        a_trainable_vars = tf.trainable_variables()
        net = a_in = tfl.input_data([None, obs_dim])
        for x in hidden_layer_top:
            net = tfl.fully_connected(net, x, activation='tanh')
        a_out = tfl.fully_connected(net, act_dim, activation='linear')
        a_trainable_vars = tf.trainable_variables()[len(a_trainable_vars):]
        return a_in, a_out, a_trainable_vars

    @staticmethod
    def create_q_net(obs_dim, act_dim, obs_in=None, act_in=None):
        hidden_layer_top = [64, 64]
        q_trainable_vars = tf.trainable_variables()

        q_in_obs = tfl.input_data([None, obs_dim]) if obs_in is None else obs_in
        q_in_act = tfl.input_data([None, act_dim]) if act_in is None else act_in

        x = hidden_layer_top[0]
        t1 = tfl.fully_connected(q_in_obs, x, activation='linear')
        t2 = tfl.fully_connected(q_in_act, x, activation='linear')
        net = tfl.activation(tf.matmul(q_in_obs, t1.W) + tf.matmul(q_in_act, t2.W) + t1.b, activation='tanh')

        for x in hidden_layer_top[1:]:
            net = tfl.fully_connected(net, x, activation='tanh')

        q_out = tfl.fully_connected(net, 1, activation='linear')
        q_trainable_vars = tf.trainable_variables()[len(q_trainable_vars):]
        return q_in_obs, q_in_act, q_out, q_trainable_vars


# Random process
class OrnsteinUhlenbeckProcess:
    def __init__(self, x0, mu, theta, sigma):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x = x0

    def next_value(self, delta=1.):
        self.x = self.x + self.theta * (self.mu - self.x) * delta + \
                 self.sigma * np.sqrt(delta) * np.random.normal(0., 1., self.x.shape)
        return self.x


# FIFO experience replay
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size=None):
        self.buffer_size = buffer_size
        self.index = 0
        self.batch_size = batch_size
        # random.sample is very slow when used with a deque
        self.buffer = []

    def add(self, s, a, r, s1):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((s, a, r, s1))
        else:
            self.buffer[self.index] = (s, a, r, s1)
            self.index += 1
            if self.index >= self.buffer_size:
                self.index = 0

    def minibatch(self, batch_size=None):
        if batch_size is None:
            if self.batch_size is None:
                raise ValueError("batch_size is None")
            batch_size = self.batch_size
        batch_size = min(batch_size, self.size())
        l = random.sample(self.buffer, batch_size)
        s = np.vstack([x[0] for x in l])
        a = np.vstack([x[1] for x in l])
        r = np.array([[x[2]] for x in l])
        s1 = np.vstack([x[3] for x in l])
        return s, a, r, s1

    def size(self):
        return len(self.buffer)


# Logger that logs whats going on while training
class Logger:
    def __init__(self):
        self.rewards = []
        self.best100avg = float('-inf')
        self.best100avgList = []
        print(' Episode |  time(s)  |  reward  |  100avg  |  bestavg  |   mean   |   var   ')

    def log(self, e, r, t):
        self.rewards.append(r)
        avg = sum(self.rewards[-100:]) / len(
            self.rewards[-100:])
        if len(self.rewards) >= 100:
            if avg > self.best100avg:
                self.best100avg = avg
        self.best100avgList.append(self.best100avg)
        m = self.rewards
        s = '{:^9}|{:^11.2f}|{:^10.2f}|{:^10.2f}|{:^11.2f}|{:^10.2f}|{:^9.2f}' \
            .format(e, t, r, avg, self.best100avg,
                    np.mean(m) if len(m) > 0 else 0, np.var(m) if len(m) > 0 else 0)
        print(s)


# The ddpg algorithm
class DDPG:
    def __init__(self, sess, env, random_process, q_lr=1e-3, a_lr=1e-4, minibatch_size=64, buffer_size=int(1e6),
                 gamma=0.99, tau=0.01):
        # define all operators
        self.sess = sess
        self.env = env
        o, a = env.observation_space, env.action_space
        self.net = Network(o.shape[0], a.shape[0])
        self.random_process = random_process
        self.replay_buffer = ReplayBuffer(buffer_size, minibatch_size)
        self.gamma = gamma

        self.r_ph = tf.placeholder(tf.float32, [None, 1])
        pred_q = self.r_ph + self.gamma * self.net.qp_out
        loss_q = tf.reduce_mean(tf.squared_difference(pred_q, self.net.q_out))
        self.opt_q = tf.train.AdamOptimizer(q_lr).minimize(loss_q, var_list=self.net.q_trainable_vars)

        loss_a = -tf.reduce_mean(self.net.q_out)
        self.opt_a = tf.train.AdamOptimizer(a_lr).minimize(loss_a, var_list=self.net.a_trainable_vars)

        self.sess.run(tf.global_variables_initializer())
        z = list(zip(self.net.q_trainable_vars + self.net.a_trainable_vars,
                     self.net.qp_trainable_vars + self.net.ap_trainable_vars))
        self.copy_nets_ops = [p.assign(tau * x + (1 - tau) * p) for x, p in z]
        self.sess.run([p.assign(x) for x, p in z])

    def train(self):
        # sample minibatch
        ms, ma, mr, ms1 = self.replay_buffer.minibatch()
        self.sess.run(self.opt_q, feed_dict={self.r_ph: mr, self.net.aqp_in: ms1,
                                             self.net.aq_in: ms, self.net.a_out: ma})
        self.sess.run(self.opt_a, feed_dict={self.net.aq_in: ms})

    def run(self):
        logger = Logger()
        episode = 0
        while True:
            sum_reward = 0.
            z = time.time()
            s = self.env.reset()[np.newaxis]
            a = self.sess.run(self.net.a_out, feed_dict={self.net.aq_in: s})
            t = False
            x = 1
            while not t:
                # run environment + save data point
                a += self.random_process.next_value()
                s1, r, t, _ = self.env.step(a[0])
                self.replay_buffer.add(s, a, r, s1[np.newaxis])

                # train both networks
                self.train()

                # update target networks
                a = self.sess.run([self.copy_nets_ops, self.net.a_out],
                                  feed_dict={self.net.aq_in: s1[np.newaxis]})[1]

                s = s1[np.newaxis]
                sum_reward += r
                x += 1
            logger.log(episode, sum_reward, time.time() - z)
            episode += 1
            if episode >= 1500:
                break
        self.env.close()
        return logger.best100avg

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    random.seed(0)

    sess = tf.Session()
    rp = OrnsteinUhlenbeckProcess(np.array([0]), 0., .15, .2)
    name = 'Swimmer-v1'
    env = gym.make(name)
    env.seed(0)
    # save_name = '/tmp/' + name + '-' + str(int(round(time.time() * 1000)))
    save_name = '/tmp/Swimmer-v1/DDPG'
    env = gym.wrappers.Monitor(env, save_name, lambda x: False)

    rl = DDPG(sess, env, rp, minibatch_size=64, q_lr=1e-3, a_lr=1e-4, buffer_size=int(1e6), gamma=.99, tau=0.001)
    rl.run()
