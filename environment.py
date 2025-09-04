import copy
import sys

import numpy as np

from grid_simulator import power_flow


def get_args(system):
    Pgmax, Pgmin, dPgmax, Umax, Umin, Pmax = [], [], [], [], [], []
    U, D, Pg, Pl, Ql = [], [], [], [], []
    a, b, c, Tu, Td, Cu = [], [], [], [], [], []
    f, t, r, x = [], [], [], []
    load_bus = []

    for bus_name, bus in system.buses.items():
        if bus.type == "PQ":
            U.append(bus.U)
            D.append(bus.D)
            Umax.append(bus.Umax)
            Umin.append(bus.Umin)
            if bus.Pl_list or bus.Ql_list:
                load_bus.append(bus.index)
                Pl.append(bus.Pl_list)
                Ql.append(bus.Ql_list)
            else:
                Pl.append([0] * (system.n_T + 1) * system.n_day)
                Ql.append([0] * (system.n_T + 1) * system.n_day)

    for bus_name, bus in system.buses.items():
        if bus.type == "PU":
            syn = system.syns[bus.syn]

            U.append(bus.U)
            D.append(bus.D)
            Umax.append(bus.Umax)
            Umin.append(bus.Umin)
            Pg.append(bus.Pg)
            if bus.Pl_list or bus.Ql_list:
                load_bus.append(bus.index)
                Pl.append(bus.Pl_list)
                Ql.append(bus.Ql_list)
            else:
                Pl.append([0] * (system.n_T + 1) * system.n_day)
                Ql.append([0] * (system.n_T + 1) * system.n_day)

            Pgmax.append(syn.Pgmax)
            Pgmin.append(syn.Pgmin)
            dPgmax.append(syn.dPgmax)
            a.append(syn.a)
            b.append(syn.b)
            c.append(syn.c)
            Tu.append(syn.Tu)
            Td.append(syn.Td)
            Cu.append(syn.Cu)

    for bus_name, bus in system.buses.items():
        if bus.type == "SW":
            syn = system.syns[bus.syn]

            U.append(bus.U)
            D.append(bus.D)
            Umax.append(bus.Umax)
            Umin.append(bus.Umin)
            Pg.append(bus.Pg)
            if bus.Pl_list or bus.Ql_list:
                load_bus.append(bus.index)
                Pl.append(bus.Pl_list)
                Ql.append(bus.Ql_list)
            else:
                Pl.append([0] * (system.n_T + 1) * system.n_day)
                Ql.append([0] * (system.n_T + 1) * system.n_day)

            Pgmax.append(syn.Pgmax)
            Pgmin.append(syn.Pgmin)
            dPgmax.append(syn.dPgmax)
            a.append(syn.a)
            b.append(syn.b)
            c.append(syn.c)
            Tu.append(syn.Tu)
            Td.append(syn.Td)
            Cu.append(syn.Cu)

    for _, line in system.lines.items():
        Pmax.append(line.Pmax)
        f.append(system.buses[line.From].index)
        t.append(system.buses[line.To].index)
        r.append(line.r)
        x.append(line.x)

    Pgmax = np.array(Pgmax, dtype=float).reshape(-1, 1)
    Pgmin = np.array(Pgmin, dtype=float).reshape(-1, 1)
    dPgmax = np.array(dPgmax, dtype=float).reshape(-1, 1)
    Umax = np.array(Umax, dtype=float).reshape(-1, 1)
    Umin = np.array(Umin, dtype=float).reshape(-1, 1)
    Pmax = np.array(Pmax, dtype=float).reshape(-1, 1)
    U = np.array(U, dtype=float).reshape(-1, 1)
    D = np.array(D, dtype=float).reshape(-1, 1)
    Pg = np.array(Pg, dtype=float).reshape(-1, 1)
    Pl = np.array(Pl).T[:, :, np.newaxis]
    Ql = np.array(Ql).T[:, :, np.newaxis]
    a = np.array(a, dtype=float).reshape(-1, 1)
    b = np.array(b, dtype=float).reshape(-1, 1)
    c = np.array(c, dtype=float).reshape(-1, 1)
    Tu = np.array(Tu, dtype=int).reshape(-1, 1)
    Td = np.array(Td, dtype=int).reshape(-1, 1)
    Cu = np.array(Cu, dtype=float).reshape(-1, 1)
    r = np.array(r, dtype=float).reshape(-1, 1)
    x = np.array(x, dtype=float).reshape(-1, 1)
    args = {"Pgmax": Pgmax, "Pgmin": Pgmin, "dPgmax": dPgmax, "Umax": Umax, "Umin": Umin, "Pmax": Pmax, "U": U, "D": D,
            "Pg": Pg, "Pl": Pl, "Ql": Ql, "a": a, "b": b, "c": c, "Tu": Tu, "Td": Td, "Cu": Cu, "load_bus": load_bus,
            "f": f, "t": t, "r": r, "x": x}
    return args


class Env:
    def __init__(self, system):
        self.system = system
        self.reset_system = copy.deepcopy(system)
        self.args = get_args(self.system)
        self.i_day = None
        self.status = None
        self.t = None

    def get_state(self):
        state = []
        state += self.status[-1]
        state += self.system.Pl.flatten().tolist()
        state += self.system.Ql.flatten().tolist()
        state += self.system.Pg.flatten().tolist()
        state += self.system.Qg.flatten().tolist()
        state += self.system.U.flatten().tolist()
        state += self.system.D.flatten().tolist()
        return state

    def get_reward(self, converge):
        if converge:
            # 平衡节点有功越限
            SW_reward = np.maximum(0, self.args["Pgmin"][-1] - self.system.Pg[-1]) / 100
            SW_reward += np.maximum(0, self.system.Pg[-1] - self.args["Pgmax"][-1]) / 100

            # 节点电压越限
            U_reward = np.maximum(0, self.args["Umin"] - self.system.U).sum()
            U_reward += np.maximum(0, self.system.U - self.args["Umax"]).sum()

            # 线路过载
            Vf_real = self.system.U[self.args["f"]] * np.cos(self.system.D[self.args["f"]])
            Vf_imag = self.system.U[self.args["f"]] * np.sin(self.system.D[self.args["f"]])
            Vt_real = self.system.U[self.args["t"]] * np.cos(self.system.D[self.args["t"]])
            Vt_imag = self.system.U[self.args["t"]] * np.sin(self.system.D[self.args["t"]])
            P_line = ((Vf_real - Vt_real) * (Vf_real * self.args["r"] - Vf_imag * self.args["x"]) + (
                    Vf_imag - Vt_imag) * (Vf_real * self.args["x"] + Vf_imag * self.args["r"])) / (
                             self.args["r"] ** 2 + self.args["x"] ** 2)
            P_reward = np.maximum(0, np.abs(P_line) - self.args["Pmax"]).sum()

            # 发电成本
            Pg = self.system.Pg * self.system.basic_S
            if self.system.Pg[-1] > 0:
                cost_reward = (self.args["a"] * Pg ** 2 + self.args["b"] * Pg + self.args["c"]).sum().item()
            else:
                cost_reward = (self.args["a"][:-1] * Pg[:-1] ** 2 + self.args["b"][:-1] * Pg[:-1] +
                               self.args["c"][:-1]).sum().item()

            # 启停成本  # TODO
            start = np.clip(np.diff(self.status[-2:], axis=0), 0, 1).reshape(-1, 1)
            cost_reward += (start * self.args["Cu"][:-1]).sum().item()

            secure_reward = (SW_reward + P_reward + U_reward).item()
            reward = -secure_reward * 100 - cost_reward * 1e-4
        else:
            reward = -(self.system.n_T + 1 - self.t) * 20
            secure_reward, shed_reward, cost_reward = 0, 0, 0
        return reward, secure_reward, cost_reward

    def reset(self):
        self.t = 0
        self.status = [[1 for _ in range(self.system.n_PU)] for _ in range(5)]

        self.system = copy.deepcopy(self.reset_system)
        self.i_day = np.random.randint(self.system.n_day)
        self.system.U = copy.deepcopy(self.args["U"])
        self.system.D = copy.deepcopy(self.args["D"])
        self.system.Pg = copy.deepcopy(self.args["Pg"])
        self.system.Pl = self.args["Pl"][:, self.args["load_bus"]][self.t + self.i_day * self.system.n_T]
        self.system.Ql = self.args["Ql"][:, self.args["load_bus"]][self.t + self.i_day * self.system.n_T]

        self.system.P = np.zeros((self.system.n_bus, 1))
        self.system.Q = np.zeros((self.system.n_bus, 1))
        self.system.P[self.args["load_bus"]] -= self.system.Pl
        self.system.Q[self.args["load_bus"]] -= self.system.Ql
        self.system.P[self.system.n_PQ:] += self.system.Pg

        converge = power_flow.NR_calculation(self.system)
        if converge:
            Pg_all = copy.deepcopy(self.system.P)
            Qg_all = copy.deepcopy(self.system.Q)
            Pg_all[self.args["load_bus"]] += self.system.Pl
            Qg_all[self.args["load_bus"]] += self.system.Ql
            self.system.Pg = Pg_all[self.system.n_PQ:]
            self.system.Qg = Qg_all[self.system.n_PQ:]
        else:
            print("PF not converge at step 0")
            sys.exit()
        state = self.get_state()
        return state

    def step(self, action):
        self.t += 1

        # TODO
        for i in range(self.system.n_PU):
            his_status = np.array(self.status).T[i]
            if his_status[-1] != action[i]:
                if action[i] == 1 and sum(his_status[-self.args["Td"][i].item():]) != 0:
                    action[i] = 0
                elif action[i] == 0 and sum(his_status[-self.args["Tu"][i].item():]) != self.args["Tu"][i]:
                    action[i] = 1
        self.status.append(action[:self.system.n_PU])

        statu = np.array(action[:self.system.n_PU]).reshape(-1, 1)
        dPg = np.array(action[self.system.n_PU:2 * self.system.n_PU]).reshape(-1, 1)
        dU = (np.array(action[2 * self.system.n_PU:]) / 2 + 0.5).reshape(-1, 1)
        self.system.Pg[:-1] += dPg * self.args["dPgmax"][:-1]
        self.system.Pg = np.clip(self.system.Pg, self.args["Pgmin"], self.args["Pgmax"])
        self.system.Pg[:-1] *= statu
        self.system.Pl = self.args["Pl"][:, self.args["load_bus"]][self.t + self.i_day * self.system.n_T]
        self.system.Ql = self.args["Ql"][:, self.args["load_bus"]][self.t + self.i_day * self.system.n_T]
        self.system.P = np.zeros((self.system.n_bus, 1))
        self.system.Q = np.zeros((self.system.n_bus, 1))
        self.system.P[self.args["load_bus"]] -= self.system.Pl
        self.system.Q[self.args["load_bus"]] -= self.system.Ql
        self.system.P[self.system.n_PQ:] += self.system.Pg
        Umin = self.args["Umin"][self.system.n_PQ:-1]
        Umax = self.args["Umax"][self.system.n_PQ:-1]
        self.system.U[self.system.n_PQ:-1] = Umin + (Umax - Umin) * dU

        converge = power_flow.NR_calculation(self.system)
        if converge:
            Pg_all = copy.deepcopy(self.system.P)
            Qg_all = copy.deepcopy(self.system.Q)
            Pg_all[self.args["load_bus"]] += self.system.Pl
            Qg_all[self.args["load_bus"]] += self.system.Ql
            self.system.Pg = Pg_all[self.system.n_PQ:]
            self.system.Qg = Qg_all[self.system.n_PQ:]
        reward, secure_reward, cost_reward = self.get_reward(converge)

        if converge:
            next_state = self.get_state()
            terminated = False
            truncated = False if self.t < self.system.n_T else True
            info = {"secure": secure_reward, "cost": cost_reward, "Pg": self.system.Pg[:-1].flatten().tolist()}
        else:
            next_state = [0] * len(self.get_state())
            terminated = True
            truncated = False
            info = "PF not converge"
        return next_state, reward, terminated, truncated, info
