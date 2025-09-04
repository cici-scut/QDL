import scipy.sparse as sp


# 节点导纳矩阵生成模块
# # #
# # #

class CSRGenerator:
    """
    稀疏生成器
    """

    def __init__(self):
        self.rows = []
        self.cols = []
        self.values = []

    def set(self, row, col, value):
        self.rows.append(row)
        self.cols.append(col)
        self.values.append(value)


def get_Y_mat(system):
    """
    生成节点导纳矩阵

    Args:
        system: 电力系统实例

    """
    Y_mat_gen = CSRGenerator()
    # 添加并联支路
    for _, bus in system.buses.items():
        if bus.g or bus.b:
            Y_mat_gen.set(bus.index, bus.index, bus.g + 1j * bus.b)

    # 添加线路
    for _, line in system.lines.items():
        From, To = system.buses[line.From].index, system.buses[line.To].index

        Y_mat_gen.set(From, From, 1 / (line.r + 1j * line.x) + 0.5j * line.b)
        Y_mat_gen.set(To, To, 1 / (line.r + 1j * line.x) + 0.5j * line.b)
        Y_mat_gen.set(From, To, -1 / (line.r + 1j * line.x))
        Y_mat_gen.set(To, From, -1 / (line.r + 1j * line.x))

    # 添加变压器
    for _, trans in system.transes.items():
        high, low = system.buses[trans.high].index, system.buses[trans.low].index

        Y_mat_gen.set(high, high, 1 / (trans.r + 1j * trans.x))
        Y_mat_gen.set(low, low, trans.tp ** 2 / (trans.r + 1j * trans.x))
        Y_mat_gen.set(high, low, -trans.tp / (trans.r + 1j * trans.x))
        Y_mat_gen.set(low, high, -trans.tp / (trans.r + 1j * trans.x))

    system.Y_mat = sp.csr_matrix((Y_mat_gen.values, (Y_mat_gen.rows, Y_mat_gen.cols)),
                                 shape=(system.n_bus, system.n_bus))
