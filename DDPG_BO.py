import numpy as np
import tensorflow as tf
from fastddpg.DDPG import DDPG, OrnsteinUhlenbeckProcess
import os
import json
import GPyOpt
import random
import gym


# function that loads previously saved data points
# i.e. a set of hyperparameter and the corresponding maximum average episode reward
def load_prev_points(path):
    dirs = [os.path.join(path, d) for d in os.listdir(path) if not os.path.isfile(os.path.join(path, d))]
    x, y = np.zeros((len(dirs), 7)), np.zeros((len(dirs), 1))
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
        x[di] = [p['theta'], p['sigma'], p['lr_p'], p['lr_v'],
                 p['mbs'], p['bc'], p['tau']]

    return x, y


# function that saves a set of hyperparameter to disk
def write_hyper_params(path, theta, sigma, lr_p, lr_v, mbs, bc, tau):
    params = {'theta': theta, 'sigma': sigma, 'lr_p': lr_p,
              'lr_v': lr_v, 'mbs': mbs, 'bc': bc, 'tau': tau}
    print(params)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path + '/hyperparams.json', 'x') as f:
        json.dump(params, f)


# function that saves the hyperparameter to disk, runs the algorithm and
# returns the maximum average episode reward with this hyperparameter set
def test(x):
    theta, sigma, lr_p, lr_v, mbs, bc, tau = x[0]
    env_name = 'Swimmer-v1'
    path = '/home/polarbart/Documents/DDPG_OAI_Test_discrete/' + env_name
    test_num = len([os.path.join(path, d) for d in os.listdir(path) if not os.path.isfile(os.path.join(path, d))])
    save_name = path + '/Test_' + str(test_num)
    write_hyper_params(save_name, theta, sigma, lr_p, lr_v, mbs, bc, tau)

    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    env = gym.make(env_name)
    env.seed(0)
    env = gym.wrappers.Monitor(env, save_name, lambda t: t == 1499)
    ou = OrnsteinUhlenbeckProcess(np.array([0, 0]), 0., theta, sigma)
    d = DDPG(tf.Session(), env, ou, lr_v, lr_p, int(mbs), int(bc), 0.999, tau)
    res = d.run()
    print(res)
    return -res


# run the hyperparameter optimization
while True:
    np.random.seed(0)
    tf.set_random_seed(0)
    random.seed(0)

    space = [{'type': 'discrete', 'domain': (0.1, 0.15, 0.2, 0.25)},  # theta
             {'type': 'discrete', 'domain': (0.15, 0.2, 0.25, 0.3)},  # sigma
             {'type': 'discrete', 'domain': (1e-4, 5e-4, 1e-3, 5e-3)},  # lr_p
             {'type': 'discrete', 'domain': (1e-4, 5e-4, 1e-3, 5e-3)},  # lr_v
             {'type': 'discrete', 'domain': (16, 32, 64, 128, 256)},  # mbs
             {'type': 'discrete', 'domain': (1e5, 2.5e5, 5e5, 7.5e5, 1e6)},  # bc
             {'type': 'discrete', 'domain': (1e-4, 5e-4, 1e-3, 5e-3, 1e-2)}]  # tau

    constrains = []

    feasible_region = GPyOpt.Design_space(space=space, constraints=constrains)

    ix, iy = load_prev_points('/home/polarbart/Documents/DDPG_OAI_Test_discrete/Swimmer-v1')
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
        bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, ix,
                                                        iy)

    bo.run_optimization(max_iter=int(1e18), eps=0.)

    print(bo.x_opt)
    print(bo.fx_opt)

