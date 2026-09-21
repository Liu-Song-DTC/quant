#!/usr/bin/env python3
"""
逆回购闲置现金sweep (实盘路径预备, 默认关闭)。

动机 (2026-09-21 census后收尾): 快照日持仓总权重均值43.1% → 平均闲置现金56.9%
在账户里零收益。闲置部分每日做1天期国债逆回购(GC001)年化≈1.5% → 约0.91% NAV/年
的无风险增益 — 比census任何被否决臂的经济梯度都大, 且零回测风险(纯实盘执行层,
回测锚点732,689不动, 实盘只会优于回测)。

机制: A股逆回购=卖出204001.SH(GC001, 上交所)/131810.SZ(R-001, 深交所),
最低1000元、1000元递增(2022-05-16改革后), T+1资金可用(可买股票)/T+2可取,
到期自动回款无需反向操作。成交价=年化收益率。执行顺序必须: 股票买单全部
成交后最后下单, 以QMT实际可用余额为准(本估算仅为上界参考)。

本模块只产估算与建议块(写入trade_orders.json的repo_sweep键), 不下真实单 —
QMT侧执行脚本不在本仓库。开关: yaml live_monitoring.repo_sweep.enabled
(live_monitoring是信号指纹豁免节, 加键零fp漂移)。
"""

REPO_CODE_GC001 = '204001.SH'   # 上交所1天期国债逆回购
REPO_CODE_R001 = '131810.SZ'    # 深交所1天期
MIN_AMOUNT = 1000               # 最低申报(元), 两市场2022后统一
INCREMENT = 1000                # 递增单位(元)
MIN_RESERVE = 1000              # 永不动用的最小缓冲(元), 防零头/手续费/滑点
SUPPORTED_TENORS = {'1d'}       # v1只做隔夜 — 7d/14d在10d调仓周期下会锁死调仓日资金


def compute_repo_sweep(post_order_cash: float, cfg: dict) -> dict:
    """估算订单执行后的闲置现金可sweep金额。

    Args:
        post_order_cash: 全部股票订单成交后预计剩余可用资金(元)
            = 现金 + 卖出回款 - 买单总额 (0h现金闸同口径)。
        cfg: yaml live_monitoring.repo_sweep 子节 ({} → 返回关闭态)。

    Returns:
        {'enabled': bool, 'code': str, 'tenor': str, 'amount': int,
         'reserve': int, 'estimated_annual_yield': float, 'reason': str}
        amount=0 且 enabled=True 表示无闲置可扫(或低于门槛), reason说明原因。
        任何输入无效(cfg关闭/负现金/tenor不支持) → enabled=False。
    """
    cfg = cfg or {}
    if not cfg.get('enabled', False):
        return {'enabled': False, 'reason': 'repo_sweep关闭'}

    tenor = str(cfg.get('tenor', '1d'))
    if tenor not in SUPPORTED_TENORS:
        # v1只支持隔夜: 更长期限在10d调仓周期下会锁死下一个调仓日的买入资金
        return {'enabled': False,
                'reason': f'tenor {tenor} 不支持(v1仅1d, 避免锁死调仓日资金)'}

    if post_order_cash is None or post_order_cash <= 0:
        return {'enabled': False, 'reason': '订单后无剩余现金'}

    reserve_ratio = float(cfg.get('reserve_ratio', 0.10))
    reserve_ratio = min(max(reserve_ratio, 0.0), 1.0)  # 钳制[0,1]
    reserve = max(post_order_cash * reserve_ratio, MIN_RESERVE)

    sweepable = post_order_cash - reserve
    if sweepable < MIN_AMOUNT:
        return {'enabled': True, 'code': REPO_CODE_GC001, 'tenor': tenor,
                'amount': 0, 'reserve': int(reserve),
                'estimated_annual_yield': 0.0,
                'reason': f'可扫{sweepable:,.0f}元 < 门槛{MIN_AMOUNT}元, 不足一笔'}

    amount = int(sweepable // INCREMENT) * INCREMENT

    assumed_rate = float(cfg.get('assumed_rate', 0.015))  # 仅估算展示用, 非成交价
    return {
        'enabled': True,
        'code': REPO_CODE_GC001,
        'tenor': tenor,
        'amount': amount,
        'reserve': int(reserve),
        'estimated_annual_yield': amount * assumed_rate,
        'reason': (f'闲置{post_order_cash:,.0f} − 缓冲{int(reserve):,} = '
                   f'{sweepable:,.0f}, 取整{amount:,} (GC001隔夜, 次日可用)'),
    }


def sweep_guidance(sweep: dict) -> list:
    """QMT侧执行指引(写入trade_orders.json的repo_sweep.notes)。"""
    if not sweep.get('enabled'):
        return []
    return [
        '执行顺序: 股票买单全部成交后, 最后下此逆回购单',
        '下单方向=卖出, 代码=%s, 数量=%d元(1000元整数倍)' % (sweep['code'], sweep['amount']),
        '以下单时QMT实际可用余额为准(本值为估算上界, 成交滑点/部分成交会使实际更少)',
        '成交价=年化利率, 市价委托即可; 15:30前下单(逆回购交易延长至15:30)',
        '到期自动回款: 次日上午资金可用(可买股票), 无需反向操作',
        '预留缓冲%d元不动, 覆盖滑点与手续费' % sweep['reserve'],
    ]
