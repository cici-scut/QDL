from pyomo.environ import *
import contextlib
import io

def DOPF_calculation(optimal_model, solver="ipopt", tee=False):
    optimal_model.U = Var(optimal_model.buses, optimal_model.hours)
    optimal_model.D = Var(optimal_model.buses, optimal_model.hours)
    optimal_model.Pg = Var(optimal_model.syns, optimal_model.hours)
    optimal_model.Qg = Var(optimal_model.syns, optimal_model.hours)

    optimal_model.obj = Objective(rule=total_cost_rule, sense=minimize)

    optimal_model.U_max_cons = Constraint(optimal_model.buses, optimal_model.hours, rule=Umax_rule)
    optimal_model.U_min_cons = Constraint(optimal_model.buses, optimal_model.hours, rule=Umin_rule)
    optimal_model.SW_U_cons = Constraint(optimal_model.hours, rule=SW_U_rule)
    optimal_model.SW_D_cons = Constraint(optimal_model.hours, rule=SW_D_rule)
    optimal_model.Pgmax_cons = Constraint(optimal_model.syns, optimal_model.hours, rule=Pgmax_rule)
    optimal_model.Pgmin_cons = Constraint(optimal_model.syns, optimal_model.hours, rule=Pgmin_rule)
    optimal_model.Pmax_cons1 = Constraint(optimal_model.lines, optimal_model.hours, rule=Pmax_rule1)
    optimal_model.Pmax_cons2 = Constraint(optimal_model.lines, optimal_model.hours, rule=Pmax_rule2)
    optimal_model.P_cons = Constraint(optimal_model.buses, optimal_model.hours, rule=P_rule)
    optimal_model.Q_cons = Constraint(optimal_model.buses, optimal_model.hours, rule=Q_rule)
    optimal_model.dP_up_cons = Constraint(optimal_model.syns, optimal_model.hours, rule=dP_up_rule)
    optimal_model.dP_down_cons = Constraint(optimal_model.syns, optimal_model.hours, rule=dP_down_rule)

    with contextlib.redirect_stdout(io.StringIO()):
        SolverFactory(solver).solve(optimal_model, tee=tee)
    if optimal_model.solutions[0]._metadata["status"] == "infeasible":
        return False
    return True


def optimal_init(system, optimal_model):
    Pgmax, Pgmin, dPgmax, Umax, Umin, Pmax = [], [], [], [], [], []
    Pg, Qg, Pl, Ql, Pl_list, Ql_list = [], [], [], [], [], []
    a, b, c, Tu, Td, Cu = [], [], [], [], [], []
    f, t, r, x = [], [], [], []

    for _, bus in system.buses.items():
        if bus.type == "PQ":
            Umax.append(bus.Umax)
            Umin.append(bus.Umin)
            Pl.append(bus.Pl)
            Ql.append(bus.Ql)
            if bus.Pl_list or bus.Ql_list:
                Pl_list.append(bus.Pl_list[1:system.n_T + 1])
                Ql_list.append(bus.Ql_list[1:system.n_T + 1])
            else:
                Pl_list.append([0] * system.n_T)
                Ql_list.append([0] * system.n_T)

    for _, bus in system.buses.items():
        if bus.type == "PU":
            syn = system.syns[bus.syn]

            Umax.append(bus.Umax)
            Umin.append(bus.Umin)
            Pg.append(bus.Pg)
            Qg.append(bus.Qg)
            Pl.append(bus.Pl)
            Ql.append(bus.Ql)
            if bus.Pl_list or bus.Ql_list:
                Pl_list.append(bus.Pl_list[1:system.n_T + 1])
                Ql_list.append(bus.Ql_list[1:system.n_T + 1])
            else:
                Pl_list.append([0] * system.n_T)
                Ql_list.append([0] * system.n_T)

            Pgmax.append(syn.Pgmax)
            Pgmin.append(syn.Pgmin)
            dPgmax.append(syn.dPgmax)
            a.append(syn.a)
            b.append(syn.b)
            c.append(syn.c)
            Tu.append(syn.Tu)
            Td.append(syn.Td)
            Cu.append(syn.Cu)

    for _, bus in system.buses.items():
        if bus.type == "SW":
            syn = system.syns[bus.syn]

            Umax.append(bus.Umax)
            Umin.append(bus.Umin)
            Pg.append(bus.Pg)
            Qg.append(bus.Qg)
            Pl.append(bus.Pl)
            Ql.append(bus.Ql)
            if bus.Pl_list or bus.Ql_list:
                Pl_list.append(bus.Pl_list[1:system.n_T + 1])
                Ql_list.append(bus.Ql_list[1:system.n_T + 1])
            else:
                Pl_list.append([0] * system.n_T)
                Ql_list.append([0] * system.n_T)

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

    optimal_model.args = {"Pgmax": Pgmax, "Pgmin": Pgmin, "Umax": Umax, "Umin": Umin, "dPgmax": dPgmax,
                          "Pmax": Pmax, "Pg": Pg, "Qg": Qg, "Pl": Pl, "Ql": Ql, "a": a, "b": b, "c": c, "Tu": Tu,
                          "Td": Td, "Cu": Cu, "from": f, "to": t, "r": r, "x": x, "Pl_list": Pl_list,
                          "Ql_list": Ql_list, "U": system.U.squeeze().tolist(), "D": system.D.squeeze().tolist()}

    optimal_model.n_T = system.n_T
    optimal_model.n_bus = system.n_bus
    optimal_model.n_PQ = system.n_PQ
    optimal_model.n_syn = system.n_syn
    optimal_model.n_line = system.n_line
    optimal_model.Y_mat = system.Y_mat.toarray().copy()

    optimal_model.buses = Set(initialize=range(system.n_bus))
    optimal_model.lines = Set(initialize=range(system.n_line))
    optimal_model.syns = Set(initialize=range(system.n_syn))
    optimal_model.hours = Set(initialize=range(system.n_T))


def total_cost_rule(model):
    total_cost = 0
    for t in range(model.n_T):
        for i in range(model.n_syn):
            Pg = model.Pg[i, t] * 100
            total_cost += model.args["a"][i] * Pg ** 2 + model.args["b"][i] * Pg + model.args["c"][i] * model.args["S"][
                i][t]
            if t:
                total_cost += model.args["S"][i][t] * (1 - model.args["S"][i][t - 1]) * model.args["Cu"][i]
    return total_cost


def Umax_rule(model, i, t):
    return model.U[i, t] <= model.args["Umax"][i]


def Umin_rule(model, i, t):
    return model.U[i, t] >= model.args["Umin"][i]


def SW_U_rule(model, t):
    return model.U[model.n_syn - 1, t] == model.args["U"][-1]


def SW_D_rule(model, t):
    return model.D[model.n_syn - 1, t] == model.args["D"][-1]


def Pgmax_rule(model, i, t):
    return model.Pg[i, t] <= model.args["Pgmax"][i] * model.args["S"][i][t]


def Pgmin_rule(model, i, t):
    return model.Pg[i, t] >= model.args["Pgmin"][i] * model.args["S"][i][t]


def Pmax_rule1(model, i, t):
    From = model.args["from"][i]
    To = model.args["to"][i]
    r = model.args["r"][i]
    x = model.args["x"][i]

    V_f_real = model.U[From, t] * cos(model.D[From, t])
    V_f_imag = model.U[From, t] * sin(model.D[From, t])
    V_t_real = model.U[To, t] * cos(model.D[To, t])
    V_t_imag = model.U[To, t] * sin(model.D[To, t])

    P = ((V_f_real - V_t_real) * (V_f_real * r - V_f_imag * x) + (V_f_imag - V_t_imag) * (
            V_f_real * x + V_f_imag * r)) / (r ** 2 + x ** 2)
    return P <= model.args["Pmax"][i]


def Pmax_rule2(model, i, t):
    From = model.args["from"][i]
    To = model.args["to"][i]
    r = model.args["r"][i]
    x = model.args["x"][i]

    V_f_real = model.U[From, t] * cos(model.D[From, t])
    V_f_imag = model.U[From, t] * sin(model.D[From, t])
    V_t_real = model.U[To, t] * cos(model.D[To, t])
    V_t_imag = model.U[To, t] * sin(model.D[To, t])

    P = ((V_f_real - V_t_real) * (V_f_real * r - V_f_imag * x) + (V_f_imag - V_t_imag) * (
            V_f_real * x + V_f_imag * r)) / (r ** 2 + x ** 2)
    return -P <= model.args["Pmax"][i]


def P_rule(model, i, t):
    sigma = 0
    for j in range(model.n_bus):
        if model.Y_mat[i, j] != 0:
            sigma += model.U[j, t] * (model.Y_mat[i, j].real * cos(model.D[i, t] - model.D[j, t]) +
                                      model.Y_mat[i, j].imag * sin(model.D[i, t] - model.D[j, t]))
    if i < model.n_PQ:
        return model.U[i, t] * sigma == -model.args["Pl_list"][i][t]
    return model.U[i, t] * sigma == model.Pg[i - model.n_PQ, t] - model.args["Pl_list"][i][t]


def Q_rule(model, i, t):
    sigma = 0
    for j in range(model.n_bus):
        if model.Y_mat[i, j] != 0:
            sigma += model.U[j, t] * (model.Y_mat[i, j].real * sin(model.D[i, t] - model.D[j, t]) -
                                      model.Y_mat[i, j].imag * cos(model.D[i, t] - model.D[j, t]))
    if i < model.n_PQ:
        return model.U[i, t] * sigma == -model.args["Ql_list"][i][t]
    return model.U[i, t] * sigma == model.Qg[i - model.n_PQ, t] - model.args["Ql_list"][i][t]


def dP_up_rule(model, i, t):
    if t:
        return model.Pg[i, t] - model.Pg[i, t - 1] <= model.args["dPgmax"][i]
    return model.Pg[i, t] - model.args["Pg"][i] <= model.args["dPgmax"][i]


def dP_down_rule(model, i, t):
    if t:
        return model.Pg[i, t - 1] - model.Pg[i, t] <= model.args["dPgmax"][i]
    return model.args["Pg"][i] - model.Pg[i, t] <= model.args["dPgmax"][i]
