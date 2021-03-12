# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
# init(context):
#     设置选股个数：20
#     划定选股范围：沪深300
#     设置回归系数
#     定义月定时器
# regression_select(context, bar_dict):
#     获取因子值
#     因子数据处理：缺失值、去极值、标准化、市值中性化
#     回归选股select_stocks(context)
#     每月调仓rebalance(context)
import numpy as np
from sklearn.linear_model import LinearRegression

def init(context):
    # 设置选股个数
    context.stock_num = 20

    # 划定选股范围
    context.hs300 = index_components("000300.XSHG")

    # 设置回归系数
    context.weights = np.array([ 0.00237299, -0.00208294, -0.0135488 ,  0.00166421,  0.00270095,
        0.00778129,  0.00037851,  0.01290236, -0.01299596])

    # 定义月定时器
    scheduler.run_monthly(regression_select, tradingday=1)

def regression_select(context, bar_dict):
    # 获取因子值
    q = query(fundamentals.eod_derivative_indicator.pe_ratio,
              fundamentals.eod_derivative_indicator.pb_ratio,
              fundamentals.eod_derivative_indicator.market_cap,
              fundamentals.financial_indicator.ev,
              fundamentals.financial_indicator.return_on_asset_net_profit,
              fundamentals.financial_indicator.du_return_on_equity,
              fundamentals.financial_indicator.earnings_per_share,
              fundamentals.income_statement.revenue,
              fundamentals.income_statement.total_expense).filter(
                  fundamentals.stockcode.in_(context.hs300))

    fund = get_fundamentals(q)
    # print("fund：\n", fund)
    # 因子数据处理：缺失值、去极值、标准化、市值中性化
    context.factor = fund.T
    # print("context.factor:\n", context.factor)
    # 处理缺失值
    context.factor = context.factor.dropna()

    # 去极值、标准化、市值中性化
    # 保留市值
    x = context.factor["market_cap"].reshape((-1, 1))

    for col in context.factor.columns:
        # 去极值
        context.factor[col] = med_method(context.factor[col])

        # 标准化
        context.factor[col] = stand_method(context.factor[col])

        y = context.factor[col]

        # 市值中性化
        if col == "market_cap":
            continue

        # 线性回归预估器流程
        estimator = LinearRegression()
        estimator.fit(x, y)
        y_predict = estimator.predict(x)
        # 市值中性化处理后的结果
        context.factor[col] = y - y_predict

    # 回归选股select_stocks(context)
    select_stocks(context)
    # 每月调仓rebalance(context)
    rebalance(context)

def select_stocks(context):
    # 经过因子处理后的结果
    # print("经过因子处理后的结果:\n", context.factor)

    # 矩阵相乘
    # 回顾矩阵相乘的方法
    # np.dot() np.matmul() matrix *
    price = np.dot(context.factor, context.weights)
    # print("price:\n", price)
    context.factor["price"] = price
    # print("context.factor:\n", context.factor)

    # 降序排列
    data = context.factor.sort_values(by="price", ascending=False)

    # 获取回归选股的列表
    context.stocks = data.index.values[:context.stock_num]

def rebalance(context):
    # 按月调仓
    # 卖出
    for stock in context.portfolio.positions.keys():
        if context.portfolio.positions[stock].quantity > 0:
            if stock not in context.stocks:
                order_target_percent(stock, 0)
    # 买入
    for stock in context.stocks:
        order_target_percent(stock, 1.0/context.stock_num)



# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    # 开始编写你的主要的算法逻辑

    # bar_dict[order_book_id] 可以拿到某个证券的bar信息
    # context.portfolio 可以拿到现在的投资组合信息

    # 使用order_shares(id_or_ins, amount)方法进行落单

    # TODO: 开始编写你的算法吧！
    # order_shares(context.s1, 1000)

    pass

# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass


# 处理特征值：去极值、标准化、市值中性化
def med_method(factor):
    # 1、找到MAD值
    med = np.median(factor)
    distance = abs(factor - med)
    MAD = np.median(distance)
    # 2、求出MAD_e
    MAD_e = 1.4826 * MAD
    # 3、求出正常值范围的边界
    up_scale = med + 3 * MAD_e
    down_scale = med - 3 * MAD_e
    # 4、替换
    factor = np.where(factor > up_scale, up_scale, factor)
    factor = np.where(factor < down_scale, down_scale, factor)
    return factor

# 自实现标准化
# (x - mean) / std
def stand_method(factor):
    mean = np.mean(factor)
    std = np.std(factor)
    factor = (factor - mean) / std
    return factor
