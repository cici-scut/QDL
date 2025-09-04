import numpy as np

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


# 潮流计算模块
# # #
# # #


def PF_init(system):
    """
    初始化潮流计算参数

    Args:
        system: 电力系统实例

    """
    # 重排节点顺序(按PQ-PU-SW的顺序排序)
    U, D, P, Q = [], [], [], []
    for _, bus in system.buses.items():
        if bus.type == "PQ":
            U.append(bus.U)
            D.append(bus.D)
            bus.P = bus.Pg
            if bus.Pl_list:
                bus.P -= bus.Pl_list[0]
                bus.Q -= bus.Ql_list[0]
            P.append(bus.P)
            Q.append(bus.Q)
    for _, bus in system.buses.items():
        if bus.type == "PU":
            U.append(bus.U)
            D.append(bus.D)
            bus.P = bus.Pg
            if bus.Pl_list:
                bus.P -= bus.Pl_list[0]
                bus.Q -= bus.Ql_list[0]
            P.append(bus.P)
            Q.append(bus.Q)
    for _, bus in system.buses.items():
        if bus.type == "SW":
            U.append(bus.U)
            D.append(bus.D)
            bus.P = bus.Pg
            if bus.Pl_list:
                bus.P -= bus.Pl_list[0]
                bus.Q -= bus.Ql_list[0]
            P.append(bus.P)
            Q.append(bus.Q)

    system.U = np.array(U, dtype=float).reshape(system.n_bus, 1)
    system.D = np.array(D, dtype=float).reshape(system.n_bus, 1)
    system.P = np.array(P, dtype=float).reshape(system.n_bus, 1)
    system.Q = np.array(Q, dtype=float).reshape(system.n_bus, 1)


def get_branch_S(system):
    """
    计算支路潮流

    Args:
        system: 电力系统实例

    """
    V = system.U * np.exp(system.D * 1j)

    # 求解线路功率
    for _, line in system.lines.items():
        f = system.buses[line.From].index
        t = system.buses[line.To].index

        V_f_real = system.U[f] * np.cos(system.D[f])
        V_f_imag = system.U[f] * np.sin(system.D[f])
        V_t_real = system.U[t] * np.cos(system.D[t])
        V_t_imag = system.U[t] * np.sin(system.D[t])

        line.Sf = ((V_f_real - V_t_real) * (V_f_real * line.r - V_f_imag * line.x) + (V_f_imag - V_t_imag) * (
                V_f_real * line.x + V_f_imag * line.r)) / (line.r ** 2 + line.x ** 2)

        # I = (V[f] - V[t]) / (line.r + 1j * line.x)
        # line.Sf = (V[f] * I.conjugate() - 1j * system.U[f] ** 2 * line.b / 2).item()
        # line.St = (V[t] * I.conjugate() - 1j * system.U[t] ** 2 * line.b / 2).item()

    # 求解变压器功率
    for _, trans in system.transes.items():
        h = system.buses[trans.high].index
        l = system.buses[trans.low].index
        I = (V[h] - V[l] * trans.tp) / (trans.r + 1j * trans.x)
        trans.Sh = (V[h] * I.conjugate()).item()
        trans.Sl = (V[l] * I.conjugate()).item()


def NR_calculation(system):
    """
    牛顿——拉夫逊法计算潮流

    Args:
        system: 电力系统实例

    Returns: 是否收敛

    """
    # 潮流计算
    k_NR = 0
    dall = None
    while k_NR < system.k_NR_max:
        k_NR += 1

        # 计算节点功率残差
        V = system.U * np.exp(system.D * 1j)
        S_now = V * (system.Y_mat * V).conj()
        dS = S_now - (system.P + 1j * system.Q)
        dG = np.row_stack((dS[:-1].real, dS[:system.n_PQ].imag))

        # 生成雅克比矩阵
        index = range(system.n_bus)
        I = system.Y_mat * V
        diagV = sp.csr_matrix((V.reshape(-1), (index, index)))
        diagI = sp.csr_matrix((I.reshape(-1), (index, index)))
        diagV_norm = sp.csr_matrix(((V / system.U).reshape(-1), (index, index)))
        dS_dU = diagV * (system.Y_mat * diagV_norm).conj() + diagI.conj() * diagV_norm  # 功率对电压幅值的偏导
        dS_dD = 1j * diagV * (diagI - system.Y_mat * diagV).conj()  # 功率对电压相角的偏导
        H = dS_dD[:-1, :-1].real
        N = dS_dU[:-1, :system.n_PQ].real
        K = dS_dD[:system.n_PQ, :-1].imag
        L = dS_dU[:system.n_PQ, :system.n_PQ].imag
        system.J_PF = sp.vstack([sp.hstack([H, N]), sp.hstack([K, L])], format="csr")

        # 更新变量
        dall = -spsolve(system.J_PF, dG).reshape(-1, 1)
        system.U[:system.n_PQ] += dall[system.n_bus - 1:, :]
        system.D[:-1] += dall[:system.n_bus - 1, :]

        # 收敛判断
        if np.max(np.abs(dall)) < system.NR_criterion or np.max(np.abs(dall)) > 100:
            if np.max(np.abs(dall)) < system.NR_criterion:
                system.P = np.array(S_now.real)
                system.Q = np.array(S_now.imag)
            break

    if k_NR == system.k_NR_max or np.max(np.abs(dall)) > 100:
        return False
    else:
        # get_branch_S(system)  # TODO
        return True


def print_PF(system, if_branch=False):
    """打印潮流计算结果"""
    print("power flow results:")
    print("============Bus data============".center(66))
    print("Name".center(8), end="")
    print("Voltage".center(20), end="")
    print("Generation".center(20), end="")
    print("Load".center(20))
    print("#".center(8), end="")
    print("U".center(10), end="")
    print("D".center(10), end="")
    print("Pg".center(10), end="")
    print("Qg".center(10), end="")
    print("Pl".center(10), end="")
    print("Ql".center(10))

    print("------".center(8), end="")
    print("-------".center(10), end="")
    print("-------".center(10), end="")
    print("-------".center(10), end="")
    print("-------".center(10), end="")
    print("-------".center(10), end="")
    print("-------".center(10))

    Pg_sum, Qg_sum, Pl_sum, Ql_sum = 0, 0, 0, 0
    for bus_name, bus in system.buses.items():
        print(bus_name.center(8), end="")
        print(("%.3f" % bus.U).center(10), end="")
        print(("%.3f" % bus.D).center(10), end="")
        if bus.type == "PU" or bus.type == "SW":
            Pg_sum += bus.Pg
            Qg_sum += bus.Qg
            print(("%.3f" % bus.Pg).center(10), end="")
            print(("%.3f" % bus.Qg).center(10), end="")
        else:
            print("-".center(10), end="")
            print("-".center(10), end="")
        if bus.Pl or bus.Ql:
            Pl_sum += bus.Pl
            Ql_sum += bus.Ql
            print(("%.3f" % bus.Pl).center(10), end="")
            print(("%.3f" % bus.Ql).center(10), end="")
        else:
            print("-".center(10), end="")
            print("-".center(10), end="")
        print()
    print("".center(28), end="")
    print("-------".center(10), end="")
    print("-------".center(10), end="")
    print("-------".center(10), end="")
    print("-------".center(10))
    print("".center(18), end="")
    print("Total:".center(10), end="")
    print(("%.3f" % Pg_sum).center(10), end="")
    print(("%.3f" % Qg_sum).center(10), end="")
    print(("%.3f" % Pl_sum).center(10), end="")
    print(("%.3f" % Ql_sum).center(10))

    if if_branch:
        print("============Line data============".center(66))
        print("Name".center(8), end="")
        print("From".center(8), end="")
        print("To".center(8), end="")
        print("P-From".center(10), end="")
        print("Q-From".center(10), end="")
        print("P-To".center(10), end="")
        print("Q-To".center(10))

        print("------".center(8), end="")
        print("------".center(8), end="")
        print("------".center(8), end="")
        print("-------".center(10), end="")
        print("-------".center(10), end="")
        print("-------".center(10), end="")
        print("-------".center(10))

        for line_name, line in system.lines.items():
            print(line_name.center(8), end="")
            print(line.From.center(8), end="")
            print(line.To.center(8), end="")
            print(("%.3f" % line.Sf.real).center(10), end="")
            print(("%.3f" % line.Sf.imag).center(10), end="")
            print(("%.3f" % line.St.real).center(10), end="")
            print(("%.3f" % line.St.imag).center(10))

        print("============Trans data============".center(66))
        print("Name".center(8), end="")
        print("High".center(8), end="")
        print("Low".center(8), end="")
        print("P-High".center(10), end="")
        print("Q-High".center(10), end="")
        print("P-Low".center(10), end="")
        print("Q-Low".center(10))

        print("------".center(8), end="")
        print("------".center(8), end="")
        print("------".center(8), end="")
        print("-------".center(10), end="")
        print("-------".center(10), end="")
        print("-------".center(10), end="")
        print("-------".center(10))

        for trans_name, trans in system.transes.items():
            print(trans_name.center(8), end="")
            print(trans.high.center(8), end="")
            print(trans.low.center(8), end="")
            print(("%.3f" % trans.Sh.real).center(10), end="")
            print(("%.3f" % trans.Sh.imag).center(10), end="")
            print(("%.3f" % trans.Sl.real).center(10), end="")
            print(("%.3f" % trans.Sl.imag).center(10))
