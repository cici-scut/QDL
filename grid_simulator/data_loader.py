import pandas as pd

from grid_simulator.system_model import *


# 数据提取模块
# # #
# # #

def get_data(system):
    """
    读取算例文件

    Args:
        system: 电力系统实例

    """
    get_bus_data(system)
    get_line_data(system)
    get_trans_data(system)
    get_syn_data(system)
    get_dyn_data(system)


def get_bus_data(system):
    """
    读取母线数据

    Args:
        system: 电力系统实例

    """
    system.buses = {}

    # 读取bus sheet信息
    bus_data = pd.read_excel(system.load_path, sheet_name="Bus", dtype={"Name": str})
    for i in range(len(bus_data)):
        if bus_data.Name[i].startswith("#"):
            continue
        system.buses[bus_data.Name[i]] = Bus(i, bus_data.Un[i], bus_data.Umax[i], bus_data.Umin[i])
        system.n_bus += 1

    # 读取Shunt sheet信息
    shunt_data = pd.read_excel(system.load_path, sheet_name="Shunt", dtype={"Name": str, "Bus": str})
    for i in range(len(shunt_data)):
        if shunt_data.Name[i].startswith("#"):
            continue
        Zb = system.buses[shunt_data.Bus[i]].Un ** 2 / system.basic_S
        system.buses[shunt_data.Bus[i]].g = shunt_data.G[i] * Zb
        system.buses[shunt_data.Bus[i]].b = shunt_data.B[i] * Zb

    # 读取PU sheet信息
    PU_data = pd.read_excel(system.load_path, sheet_name="PU", dtype={"Name": str, "Bus": str})
    for i in range(len(PU_data)):
        if PU_data.Name[i].startswith("#"):
            continue
        system.buses[PU_data.Bus[i]].type = "PU"
        system.buses[PU_data.Bus[i]].Pg = PU_data.P[i]
        system.buses[PU_data.Bus[i]].U = PU_data.U[i]
        system.n_PU += 1

    system.n_PQ = system.n_bus - system.n_PU - 1

    # 读取SW sheet信息
    SW_data = pd.read_excel(system.load_path, sheet_name="SW", dtype={"Name": str, "Bus": str})
    for i in range(len(SW_data)):
        if SW_data.Name[i].startswith("#"):
            continue
        system.buses[SW_data.Bus[i]].type = "SW"
        system.buses[SW_data.Bus[i]].U = SW_data.U[i]
        system.buses[SW_data.Bus[i]].D = SW_data.D[i]
        break

    # 按PQ-PV-SW的顺序构建母线索引
    Index = 0
    for _, bus in system.buses.items():
        if bus.type == "PQ":
            bus.index = Index
            Index += 1
    for _, bus in system.buses.items():
        if bus.type == "PU":
            bus.index = Index
            Index += 1
    for _, bus in system.buses.items():
        if bus.type == "SW":
            bus.index = Index


def get_line_data(system):
    """
    读取线路数据

    Args:
        system: 电力系统实例

    """
    system.lines = {}
    line_data = pd.read_excel(system.load_path, sheet_name="Line", dtype={"Name": str, "From": str, "To": str})
    for i in range(len(line_data)):
        if line_data.Name[i].startswith("#"):
            continue

        system.lines[line_data.Name[i]] = Line(i, line_data.From[i], line_data.To[i], line_data.R[i], line_data.X[i],
                                               line_data.B[i], line_data.Pmax[i], line_data.Qmax[i])
        system.n_line += 1


def get_trans_data(system):
    """
    读取变压器数据

    Args:
        system: 电力系统实例

    """
    system.transes = {}
    trans_data = pd.read_excel(system.load_path, sheet_name="Trans", dtype={"Name": str, "From": str, "To": str})
    for i in range(len(trans_data)):
        if trans_data.Name[i].startswith("#"):
            continue

        # 归算到高压侧
        bus_from, bus_to = trans_data.From[i], trans_data.To[i]
        U_from, U_to = system.buses[bus_from].Un, system.buses[bus_to].Un
        if U_from >= U_to:
            bus_high, bus_low = bus_from, bus_to
            U_high, U_low = U_from, U_to
            tap_ratio = trans_data.Tp[i]
        else:
            bus_high, bus_low = bus_to, bus_from
            U_high, U_low = U_to, U_from
            tap_ratio = 1 / trans_data.Tp[i]

        system.transes[trans_data.Name[i]] = Trans(i + system.n_line, bus_high, bus_low, U_high, U_low, trans_data.R[i],
                                                   trans_data.X[i], tap_ratio, trans_data.Pmax[i], trans_data.Qmax[i])
        system.n_trans += 1


def get_syn_data(system):
    """
    读取发电机数据

    Args:
        system: 电力系统实例

    """
    system.syns = {}
    syn_data = pd.read_excel(system.load_path, sheet_name="Syn", dtype={"Name": str, "Bus": str})
    for i in range(len(syn_data)):
        if syn_data.Name[i].startswith("#"):
            continue

        system.syns[syn_data.Name[i]] = Syn(syn_data.Bus[i], syn_data.Pgmax[i], syn_data.Pgmin[i], syn_data.dPgmax[i],
                                            syn_data.a[i], syn_data.b[i], syn_data.c[i], syn_data.Tu[i],
                                            syn_data.Td[i], syn_data.Cu[i])
        system.buses[syn_data.Bus[i]].syn = syn_data.Name[i]
        system.n_syn += 1


def get_dyn_data(system):
    """
    读取时序数据

    Args:
        system: 电力系统实例

    """
    dyn_Pl_data = pd.read_excel(system.load_path, sheet_name="Pl", header=None, dtype=str)
    dyn_Ql_data = pd.read_excel(system.load_path, sheet_name="Ql", header=None, dtype=str)

    for i in range(dyn_Pl_data.shape[1] - 1):
        system.buses[dyn_Pl_data[i + 1][0]].Pl_list = dyn_Pl_data[i + 1][1:].astype(float).tolist()
        system.buses[dyn_Ql_data[i + 1][0]].Ql_list = dyn_Ql_data[i + 1][1:].astype(float).tolist()
