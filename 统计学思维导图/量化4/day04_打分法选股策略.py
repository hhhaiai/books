# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
# 打分法选股逻辑？
#     如果要处理的股票过多，就分组打分
#         分成10组
#         第一组 得1分
#         第二组 得2分
#         ……
#         第十组 得1分
# 选股区间
#     2010年1月1日~2018年1月1日
# 调仓周期
#     按月调仓
#     选20只股票
# 策略主体：
#     init
#         设置选股的数量：20
#         设置分组的组数：10
#         设置月定时器
#     score_select(context, bar_dict):
#         获取因子值
#             处理缺失值
#             需不需要进行去极值、标准化、市值中性化？
#                 不需要
#                 因子打分法选股只需要知道因子之间的排名状况
#                 而去极值、标准化、市值中性化都不会影响排名
#         分组打分
#             select_stocks(context)
#         按月调仓
#             rebalance(context)
import pandas as pd

def init(context):
    # 设置选股的数量：20
    context.stock_num = 20

    # 设置分组的组数：10
    context.group_num = 10

    # 设置月定时器
    scheduler.run_monthly(score_select, tradingday=1)

def score_select(context, bar_dict):
    # 获取因子值
    # 实例化query对象
    # 因子升序
    # 因子值越小越好
    # 市值-market_cap、市盈率-pe_ratio、市净率-pb_ratio
    # 因子降序
    # 因子值越大越好
    # ROIC-return_on_invested_capital、inc_revenue-营业总收入 和inc_profit_before_tax-利润增长率
    q = query(fundamentals.eod_derivative_indicator.market_cap,
    fundamentals.eod_derivative_indicator.pe_ratio,
    fundamentals.eod_derivative_indicator.pb_ratio,
    fundamentals.financial_indicator.return_on_invested_capital,
    fundamentals.financial_indicator.inc_revenue,
    fundamentals.financial_indicator.inc_profit_before_tax)

    fund = get_fundamentals(q)
    # print("返回的结果：\n", fund)
    # 转置
    factor = fund.T
    # 缺失值处理
    context.factor = factor.dropna()
    # 分组打分
    select_stocks(context)
    # 每月调仓
    rebalance(context)

def select_stocks(context):
    # 分组打分
    # print("上一步因子处理的结果：\n", context.factor)

    # 分两种情况排序
    for col in context.factor.columns:
        # print(col, type(col))
        if col in ["market_cap", "pe_ratio", "pb_ratio"]:
            # 因子升序
            data = context.factor.sort_values(by=col)[col]
        else:
            # 因子降序
            data = context.factor.sort_values(by=col, ascending=False)[col]
            # print(data)

        # 为了方便分组打分，把Series转换成DataFrame
        data = pd.DataFrame(data)
        # 添加一列打分列
        data[col+"_score"] = 0
        # 求出每组的股票数量
        num = len(data) // context.group_num
        # 分组打分0~9
        for i in range(context.group_num):
            # 最后一组跟前面9组情况不一样
            if i == context.group_num - 1:

                data[col+"_score"][i*num:] = i+1

            else:

                data[col+"_score"][i*num: (i+1)*num] = i+1
        # print(data)
        # 合并处理
        context.factor = pd.concat([context.factor, data[col+"_score"]], axis=1)
    # print("合并处理的结果：\n", context.factor)

    # 计算打分的总分
    # print(context.factor.iloc[:, -6:].sum(axis=1).sort_values()[:context.stock_num])
    # 保留选股结果
    context.stocks = context.factor.iloc[:, -6:].sum(axis=1).sort_values()[:context.stock_num].index.values

def rebalance(context):
    # 每月调仓
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
