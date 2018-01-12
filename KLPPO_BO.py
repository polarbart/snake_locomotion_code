import numpy as np
import tensorflow as tf
from ppo.PPODMT2 import learn
import os
import json
import gym
import GPyOpt
import random

# function that loads previously saved data points
# i.e. a set of hyperparameter and the corresponding maximum average episode reward
def load_prev_points(path):
    dirs = [os.path.join(path, d) for d in os.listdir(path) if not os.path.isfile(os.path.join(path, d))]
    x, y = np.zeros((len(dirs), 9)), np.zeros((len(dirs), 1))
    for di, d in enumerate(dirs):
        fs = os.listdir(d)

        with open(os.path.join(d, next(x for x in fs if 'stats' in x)), 'r') as f:
            rewards = json.load(f)['episode_rewards']
        y[di] = float('-inf')
        for i in range(len(rewards) - 100):
            y[di] = max(y[di], np.mean(rewards[i:i + 100]))
        y[di] = -y[di]
        with open(os.path.join(d, next(x for x in fs if 'hyper' in x)), 'r') as f:
            p = json.load(f)
        p['h'], p['tkl'], p['l'] = p.pop('horizon'), p.pop('target_kl'), p.pop(
            'lambda')
        p['mbs_p'], p['mbs_v'] = min(p['mbs_p'], p['h']), min(p['mbs_v'], p['h'])
        x[di] = [p['h'], p['tkl'], p['ep_p'], p['ep_v'], p['lr_p'], p['lr_v'], p['mbs_p'], p['mbs_v'], p['l']]

    return x, y


# function that saves a set of hyperparameter to disk
def write_hyper_params(path, h, tkl, ep_p, ep_v, lr_p, lr_v, mbs_p, mbs_v, l):
    params = {'horizon': h, 'target_kl': tkl, 'ep_p': ep_p, 'ep_v': ep_v,
              'lr_p': lr_p, 'lr_v': lr_v, 'mbs_p': mbs_p, 'mbs_v': mbs_v, 'lambda': l}
    print(params)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path + '/hyperparams.json', 'x') as f:
        json.dump(params, f)


# function that saves the hyperparameter to disk, runs the algorithm and
# returns the maximum average episode reward with this hyperparameter set
def test(x):
    print(x)
    # env, sess, horizon, eta, target_kl, ep_p, ep_v, lr_p, lr_v, mbs_p, mbs_v, gamma, lam,
    # alpha, beta_low, beta_high
    h, tkl, ep_p, ep_v, lr_p, lr_v, mbs_p, mbs_v, l = x[0]
    env_name = 'Swimmer-v1'
    path = '/home/polarbart/Documents/PPODM_OAI_Test_discrete/' + env_name
    test_num = len([os.path.join(path, d) for d in os.listdir(path) if not os.path.isfile(os.path.join(path, d))])
    save_name = path + '/Test_' + str(test_num)
    write_hyper_params(save_name, h, tkl, ep_p, ep_v, lr_p, lr_v, mbs_p, mbs_v, l)

    np.random.seed(0)
    tf.set_random_seed(0)

    env = gym.make(env_name)
    env.seed(0)
    env = gym.wrappers.Monitor(env, save_name, lambda t: t == 1499)

    res = learn(env, tf.Session(), int(h), 50., tkl, int(ep_p), int(ep_v), lr_p, lr_v, int(mbs_p), int(mbs_v), 1., l, 1.5, .5, 2.)
    print(res)
    return -res

# run the hyperparameter optimization
while True:
    np.random.seed(0)
    tf.set_random_seed(0)
    random.seed(0)

    space = [{'type': 'discrete', 'domain': (256, 512, 1024, 2048, 4096, 8192)},  # h
             {'type': 'discrete', 'domain': (0.0008, 0.001, 0.003, 0.005, 0.007, 0.01)},  # tkl
             {'type': 'discrete', 'domain': (4, 8, 12, 16, 20)},  # ep_p
             {'type': 'discrete', 'domain': (4, 8, 12, 16, 20)},  # ep_v
             {'type': 'discrete', 'domain': (1e-4, 5e-4, 1e-3, 5e-3)},  # lr_p
             {'type': 'discrete', 'domain': (1e-4, 5e-4, 1e-3, 5e-3)},  # lr_v
             {'type': 'discrete', 'domain': (256, 512, 1024, 2048)},  # mbs_p
             {'type': 'discrete', 'domain': (256, 512, 1024, 2048)},  # mbs_v
             {'type': 'discrete', 'domain': (0.85, 0.9, 0.95, 0.98)}]  # l

    constrains = [{'name': 'const_1', 'constrain': 'x[:,6]-x[:,0]'},
                  {'name': 'const_2', 'constrain': 'x[:,7]-x[:,0]'}]

    feasible_region = GPyOpt.Design_space(space=space, constraints=constrains)

    ix, iy = load_prev_points('/home/polarbart/Documents/PPODM_OAI_Test_discrete/Swimmer-v1')
    if len(ix) == 0:
        ix = GPyOpt.util.stats.initial_design('random', feasible_region, 5)
        print('ok')
    objective = GPyOpt.core.task.SingleObjective(test)
    model = GPyOpt.models.GPModel()
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region, current_X=ix)
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    bo = None
    if len(iy) == 0:
        bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, ix)
    else:
        bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, ix, iy)

    bo.run_optimization(max_iter=int(1e18), eps=0.)

    print(bo.x_opt)
    print(bo.fx_opt)
