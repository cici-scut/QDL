import copy
import multiprocessing as mp
import os
import random
import time

import numpy as np
from matplotlib import pyplot as plt
import pyomo.environ as pe

import utils
from grid_simulator import system_model, data_loader, admit, power_flow, dynamic_acopf

# hyperparameter
feature_ranges = [(20, 50), (0.1, 0.2)]
n_bins = 10
n_episode = 100000
sigma = 10
n_worker = 10
pop_size = 2
lr = 0.05
performance_bound = [-150, -100]

# environment
load_path = "data/case30.xlsx"
system = system_model.System(load_path)
data_loader.get_data(system)
admit.get_Y_mat(system)
power_flow.PF_init(system)
power_flow.NR_calculation(system)
model = pe.ConcreteModel()
dynamic_acopf.optimal_init(system, model)

state = np.array(model.args["Pl_list"]).flatten().tolist()  # change according to specific energy profile
dim_state = len(state)
dim_action = system.n_PU * system.n_T

# some settings
utils.set_random_seed(345)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
util = np.maximum(0, np.log(pop_size / 2 + 1) - np.log(np.arange(1, pop_size + 1)))
utility = util / util.sum() - 1 / pop_size


class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t = 0
        self.lr = lr
        self.params = params
        self.exp_avg = np.zeros_like(self.params)
        self.exp_avg_sq = np.zeros_like(self.params)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grads = None

    def step(self):
        self.t += 1
        self.exp_avg = self.exp_avg * self.beta1 + (1 - self.beta1) * self.grads
        self.exp_avg_sq = self.exp_avg_sq * self.beta2 + (1 - self.beta2) * self.grads ** 2
        self.params -= self.lr / (1 - self.beta1 ** self.t) * self.exp_avg / (
                np.sqrt(self.exp_avg_sq) / np.sqrt(1 - self.beta2 ** self.t) + self.eps)


class Individual:
    def __init__(self):
        self.shapes = [(dim_state, 128), (128, 128), (128, dim_action)]
        self.params = np.random.randn(sum(shape[0] * shape[1] + shape[1] for shape in self.shapes))
        self.reshaped_params = []

        self.model = copy.deepcopy(model)

        self.statu = None
        self.Pg = None

        self.features = None
        self.indexes = None
        self.fitness = None

    def reshape_params(self):
        self.reshaped_params, i = [], 0
        for shape in self.shapes:
            n_w, n_b = shape[0] * shape[1], shape[1]
            self.reshaped_params += [self.params[i: i + n_w].reshape(shape),
                                     self.params[i + n_w: i + n_w + n_b].reshape((1, shape[1]))]
            i += n_w + n_b

    def get_feature(self):
        start_count = np.sum(np.diff(self.statu) == 1)
        total_statu = np.vstack((self.statu, [1] * system.n_T))
        load_rate = (self.Pg / np.array(self.model.args["Pgmax"])[:, None] * total_statu).sum() / total_statu.sum()
        self.features = [start_count, load_rate]
        return self.features

    def get_index(self):
        indexes = []
        for i, feature in enumerate(self.features):
            lower, upper = feature_ranges[i]
            norm_feature = (feature - lower) / (upper - lower)
            index = int(norm_feature * n_bins)
            index = max(0, min(index, n_bins - 1))
            indexes.append(index)
        self.indexes = tuple(indexes)
        return self.indexes

    def repair(self):
        for i in range(system.n_PU):
            t = 0
            while t < system.n_T:
                duration = 1
                while t + duration < system.n_T and self.statu[i, t + duration] == self.statu[i, t]:
                    duration += 1
                required_duration = self.model.args["Tu"][i] if self.statu[i, t] == 1 else self.model.args["Td"][i]
                if t + duration < system.n_T and duration < required_duration:
                    self.statu[i, t:t + required_duration] = self.statu[i, t]
                    t = 0
                else:
                    t += duration

    def get_fitness(self):
        self.reshape_params()
        self(np.array(state))
        self.model.args["S"] = np.vstack([self.statu, [1] * system.n_T]).tolist()
        converge = dynamic_acopf.DOPF_calculation(self.model)
        if converge:
            self.fitness = -pe.value(self.model.obj) / 10
            self.Pg = [[self.model.Pg[i, t].value for t in range(system.n_T)] for i in range(system.n_syn)]
        else:
            self.fitness = -2000
            self.Pg = [[0 for _ in range(system.n_T)] for _ in range(system.n_syn)]
        return self.fitness

    def __call__(self, x):
        x = x.reshape(1, -1)
        for i in range(len(self.shapes) - 1):
            x = x.dot(self.reshaped_params[2 * i]) + self.reshaped_params[2 * i + 1]
            x = np.tanh(x)
        x = x.dot(self.reshaped_params[-2]) + self.reshaped_params[-1]
        self.statu = (x > 0).reshape(system.n_PU, system.n_T)
        self.repair()

class Worker(mp.Process):
    def __init__(self, i, map, i_episode, seed):
        super(Worker, self).__init__()
        self.i = i
        self.map = map
        self.i_episode = i_episode
        self.seed = seed

    def run(self):
        utils.set_random_seed(self.seed)
        while self.i_episode.value < n_episode:
            noise_seed = np.random.randint(0, int(1e8), size=pop_size // 2).repeat(2).tolist()

            parent = Individual()
            parent.params = self.map[random.choice(list(self.map.keys()))].params
            optimizer = Adam(parent.params, lr)
            children, fitness_list = [], []
            for i in range(pop_size):
                child = Individual()
                np.random.seed(noise_seed[i])
                child.params = parent.params + 2 * (i % 2 - 0.5) * sigma * np.random.randn(child.params.size)
                children.append(child)

                fitness = child.get_fitness()
                child.get_feature()
                index = child.get_index()
                if index not in self.map.keys() or fitness > self.map[index].fitness:
                    self.map[index] = child

                fitness_list.append(fitness)

            grads = np.zeros_like(parent.params)
            ranks = np.argsort(fitness_list)[::-1].tolist()
            for i, rank in enumerate(ranks):
                np.random.seed(noise_seed[rank])
                grads += utility[i] * 2 * (rank % 2 - 0.5) * np.random.randn(parent.params.size)
            optimizer.grads = -grads / (pop_size * sigma)
            optimizer.step()

            fitness = parent.get_fitness()
            parent.get_feature()
            index = parent.get_index()
            if index not in self.map.keys() or fitness > self.map[index].fitness:
                self.map[index] = parent

            self.i_episode.value += 1


def process_individual(_):
    individual = Individual()
    individual.get_fitness()
    individual.get_feature()
    individual.get_index()
    return individual


if __name__ == "__main__":
    # initialization
    start = time.time()
    i_episode = mp.Value("i", 0)

    map = {}
    with mp.Pool() as pool:
        results = pool.map(process_individual, range(n_bins ** len(feature_ranges)))
    for individual in results:
        if individual.indexes not in map.keys() or individual.fitness > map[individual.indexes].fitness:
            map[individual.indexes] = individual
    map = mp.Manager().dict(map)
    fitness_list = utils.implot(map, n_bins, feature_ranges, performance_bound, "start_count", "load_rate")
    sum_fitness = np.sum(fitness_list)
    print("episode %d, n_cell %d, fitness (mean %.1f, max %.1f, sum %.1f), time %.1f" % (
        0, len(map), np.mean(fitness_list), np.max(fitness_list), sum_fitness,
        time.time() - start))
    print("initialization cost %.2f" % (time.time() - start))

    # start worker
    workers = []
    for i in range(n_worker):
        worker = Worker(i, map, i_episode, random.randint(0, int(1e8)))
        worker.start()
        workers.append(worker)

    start = time.time()
    sum_fitness_list = []
    while True:
        current_episode = i_episode.value
        if current_episode > n_episode:
            break
        if not (current_episode + 1) % 100:
            fitness_list = utils.implot(map, n_bins, feature_ranges, performance_bound, "start_count", "load_rate")
            sum_fitness = np.sum(fitness_list)
            sum_fitness_list.append(sum_fitness)
            print("episode %d, n_cell %d, fitness (mean %.1f, max %.1f, sum %.1f), time %.1f" % (
                current_episode + 1, len(map), np.mean(fitness_list), np.max(fitness_list), sum_fitness,
                time.time() - start))
            time.sleep(10)

    for worker in workers:
        worker.kill()

    utils.plot("episode", "reward", sum_fitness_list, title="MAP-elite reward")
    np.save("map1.npy", np.array(dict(map)))
