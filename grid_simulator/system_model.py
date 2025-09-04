# 系统类和设备类
# # #
# # #


class System:
    def __init__(self, load_path):
        self.load_path = load_path  # 数据文件地址
        self.basic_S = 100  # 系统的功率基值

        # 潮流计算参数
        self.k_NR_max = 20  # 牛顿—拉夫逊法最大允许迭代次数
        self.NR_criterion = 1e-6  # 牛顿—拉夫逊法收敛判据

        # 设备数量
        self.n_bus = 0  # 总节点数量
        self.n_PQ = 0  # PQ节点数量
        self.n_PU = 0  # PU节点数量
        self.n_line = 0  # 线路数量
        self.n_trans = 0  # 变压器数量
        self.n_syn = 0  # 发电机数量

        # 设备字典
        self.buses = None  # 母线字典
        self.lines = None  # 线路字典
        self.transes = None  # 变压器字典
        self.syns = None  # 发电机字典

        # 系统矩阵
        self.J_PF = None  # 潮流计算的雅克比矩阵
        self.Y_mat = None  # 静态节点导纳矩阵

        # 系统参数列向量
        self.U = None  # 节点电压幅值
        self.D = None  # 节点电压相角
        self.P = None  # 节点注入有功
        self.Q = None  # 节点注入无功
        self.Pg = None  # 节点有功出力
        self.Qg = None  # 节点无功出力
        self.Pl = None  # 节点有功负荷
        self.Ql = None  # 节点无功负荷

        self.n_T = 24  # 时间步数  # TODO
        self.n_day = 1  # 天数  # TODO
        self.optimal_model = None  # 优化模型


class Bus:
    def __init__(self, index, Un, Umax, Umin):
        self.index = index  # 索引
        self.Un = Un  # 电压基值
        self.Umax = Umax  # 电压上限
        self.Umin = Umin  # 电压下限
        self.type = "PQ"  # 类型
        self.U = 1.  # 电压
        self.D = 0.  # 相角
        self.P = 0.  # 有功
        self.Q = 0.  # 无功
        self.Pl = 0.  # 有功负荷
        self.Ql = 0.  # 无功负荷
        self.Pg = 0.  # 有功出力
        self.Qg = 0.  # 无功出力
        self.g = 0.  # 并联电导
        self.b = 0.  # 并联电纳

        self.syn = None  # 所连发电机名
        self.Pl_list = None  # 时序有功负荷
        self.Ql_list = None  # 时序无功负荷


class Line:
    def __init__(self, index, From, To, r, x, b, Pmax, Qmax):
        self.index = index  # 索引
        self.From = From  # 始端母线名
        self.To = To  # 末端母线名
        self.r = r  # 电阻
        self.x = x  # 电抗
        self.b = b  # 充电电容电纳
        self.Sf = 0  # 传输功率(首端母线注入)
        self.St = 0  # 传输功率(末端母线注入)
        self.Pmax = Pmax  # 有功上限
        self.Qmax = Qmax  # 无功上限


class Trans:
    def __init__(self, index, high, low, Un_high, Un_low, r, x, tp, Pmax, Qmax):
        self.index = index  # 索引
        self.high = high  # 高压侧母线名
        self.low = low  # 低压侧母线名
        self.Un_high = Un_high  # 高压侧电压
        self.Un_low = Un_low  # 低压侧电压
        self.r = r  # 电阻
        self.x = x  # 电抗
        self.tp = tp  # 变比(标幺值)
        self.Sh = 0  # 传输功率(高压侧母线注入)
        self.Sl = 0  # 传输功率(低压侧母线注入)
        self.Pmax = Pmax  # 传输的有功上限
        self.Qmax = Qmax  # 传输的无功上限


class Syn:
    def __init__(self, bus, Pgmax, Pgmin, dPgmax, a, b, c, Tu, Td, Cu):
        self.bus = bus  # 机端母线名

        # 发电机上下限
        self.Pgmax = Pgmax  # 有功出力上限
        self.Pgmin = Pgmin  # 有功出力下限
        self.dPgmax = dPgmax  # 有功爬坡上限

        # 发电机耗量参数
        self.a = a  # 耗量特性二次项系数
        self.b = b  # 耗量特性一次项系数
        self.c = c  # 耗量特性常数项

        self.Tu = Tu  # 最小连续开启时间
        self.Td = Td  # 最小连续关停时间
        self.Cu = Cu  # 启动成本
