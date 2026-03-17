# core/industry_factor_config.py
"""
行业因子配置模块 - 基于分行业因子验证结果

从 factor_validation_results/all_results.csv 中提取各行业Top因子
配置结构: {行业名: {factors: [因子列表], direction: {因子名: 方向}}}
direction: 1=正向(因子值高买), -1=反向(因子值低买)
"""

# 基于因子验证结果的行业最优因子配置
# Top因子来源: strategy/factor_validation_results/all_results.csv
INDUSTRY_FACTOR_CONFIG = {
    # === 电力设备 ===
    '电力设备': {
        'factors': [
            'fund_score',          # IC=7.79%, IR=0.38
            'fund_profit_growth',  # IC=5.85%, IR=0.32
            'fund_roe',           # IC=5.64%, IR=0.28
            'mom_x_lowvol_20_20', # IC=6.22%, IR=0.25
            'mom_x_lowvol_20_10', # IC=6.15%, IR=0.25
        ],
        'direction': {
            'fund_profit_growth': 1,
            'fund_score': 1,
            'fund_roe': 1,
            'mom_x_lowvol_20_20': 1,
            'mom_x_lowvol_20_10': 1,
        },
        'weights': [0.25, 0.25, 0.20, 0.15, 0.15],
    },

    # === 新能源车/风电 ===
    '新能源车/风电': {
        'factors': [
            'fund_profit_growth',  # IC=7.82%, IR=0.34
            'fund_revenue_growth', # IC=5.86%, IR=0.25
            'fund_score',          # IC=5.87%, IR=0.25
            'fund_roe',           # IC=3.84%, IR=0.16
        ],
        'direction': {
            'fund_profit_growth': 1,
            'fund_revenue_growth': 1,
            'fund_score': 1,
            'fund_roe': 1,
        },
        'weights': [0.35, 0.25, 0.25, 0.15],
    },

    # === 互联网/软件 ===
    '互联网/软件': {
        'factors': [
            'mom_x_lowvol_20_10', # IC=8.17%, IR=0.32
            'mom_x_lowvol_20_20', # IC=8.15%, IR=0.31
            'fund_profit_growth', # IC=6.83%, IR=0.25
            'bb_rsi_combo',       # IC=5.04%, IR=0.19
            'rsi_vol_combo',      # IC=4.97%, IR=0.19
        ],
        'direction': {
            'mom_x_lowvol_20_10': 1,
            'mom_x_lowvol_20_20': 1,
            'fund_profit_growth': 1,
            'bb_rsi_combo': 1,
            'rsi_vol_combo': 1,
        },
        'weights': [0.30, 0.25, 0.20, 0.15, 0.10],
    },

    # === 基建/地产/石油石化 ===
    '基建/地产/石油石化': {
        'factors': [
            'fund_profit_growth', # IC=7.24%, IR=0.28
            'fund_roe',           # IC=6.86%, IR=0.25
            'fund_score',         # IC=5.62%, IR=0.20
            'fund_eps',           # IC=4.26%, IR=0.14
        ],
        'direction': {
            'fund_profit_growth': 1,
            'fund_roe': 1,
            'fund_score': 1,
            'fund_eps': 1,
        },
        'weights': [0.30, 0.30, 0.25, 0.15],
    },

    # === 有色/钢铁/煤炭/建材 ===
    '有色/钢铁/煤炭/建材': {
        'factors': [
            'fund_score',          # IC=5.44%, IR=0.27
            'fund_profit_growth', # IC=4.76%, IR=0.26
            'fund_roe',           # IC=4.50%, IR=0.21
            'fund_revenue_growth',# IC=3.39%, IR=0.20
            'fund_eps',           # IC=3.93%, IR=0.19
        ],
        'direction': {
            'fund_score': 1,
            'fund_profit_growth': 1,
            'fund_roe': 1,
            'fund_revenue_growth': 1,
            'fund_eps': 1,
        },
        'weights': [0.25, 0.25, 0.20, 0.15, 0.15],
    },

    # === 自动化/制造 ===
    '自动化/制造': {
        'factors': [
            'fund_score',          # IC=7.17%, IR=0.25
            'fund_profit_growth', # IC=6.31%, IR=0.23
            'fund_roe',           # IC=4.67%, IR=0.15
            'fund_revenue_growth',# IC=4.00%, IR=0.14
            'mom_x_lowvol_20_20', # IC=4.06%, IR=0.13
        ],
        'direction': {
            'fund_score': 1,
            'fund_profit_growth': 1,
            'fund_roe': 1,
            'fund_revenue_growth': 1,
            'mom_x_lowvol_20_20': 1,
        },
        'weights': [0.30, 0.25, 0.20, 0.15, 0.10],
    },

    # === 电子 ===
    '电子': {
        'factors': [
            'fund_profit_growth', # IC=5.07%, IR=0.25
            'fund_score',         # IC=4.79%, IR=0.19
            'fund_roe',          # IC=4.76%, IR=0.18
            'fund_revenue_growth',# IC=3.99%, IR=0.19
            'mom_x_lowvol_20_20', # IC=4.82%, IR=0.20
            'mom_x_lowvol_20_10', # IC=4.73%, IR=0.20
        ],
        'direction': {
            'fund_profit_growth': 1,
            'fund_score': 1,
            'fund_roe': 1,
            'fund_revenue_growth': 1,
            'mom_x_lowvol_20_20': 1,
            'mom_x_lowvol_20_10': 1,
        },
        'weights': [0.25, 0.20, 0.18, 0.17, 0.10, 0.10],
    },

    # === 半导体/光伏 ===
    '半导体/光伏': {
        'factors': [
            'mom_x_lowvol_20_20', # IC=6.21%, IR=0.22
            'mom_x_lowvol_20_10', # IC=5.79%, IR=0.21
            'bb_rsi_combo',       # IC=5.04%, IR=0.18
            'momentum_reversal',  # IC=5.07%, IR=0.18
            'rsi_vol_combo',      # IC=4.86%, IR=0.17
            'fund_score',         # IC=3.75%, IR=0.13
        ],
        'direction': {
            'mom_x_lowvol_20_20': 1,
            'mom_x_lowvol_20_10': 1,
            'bb_rsi_combo': 1,
            'momentum_reversal': 1,
            'rsi_vol_combo': 1,
            'fund_score': 1,
        },
        'weights': [0.25, 0.20, 0.18, 0.15, 0.12, 0.10],
    },

    # === 通信/计算机 ===
    '通信/计算机': {
        'factors': [
            'fund_score',          # IC=5.78%, IR=0.21
            'fund_profit_growth', # IC=5.49%, IR=0.20
            'fund_roe',           # IC=4.67%, IR=0.17
            'fund_revenue_growth',# IC=4.68%, IR=0.17
            'fund_eps',           # IC=4.45%, IR=0.16
        ],
        'direction': {
            'fund_score': 1,
            'fund_profit_growth': 1,
            'fund_roe': 1,
            'fund_revenue_growth': 1,
            'fund_eps': 1,
        },
        'weights': [0.28, 0.25, 0.20, 0.15, 0.12],
    },

    # === 军工 ===
    '军工': {
        'factors': [
            'fund_profit_growth', # IC=7.00%, IR=0.20
            'mom_x_lowvol_20_20', # IC=6.88%, IR=0.17
            'mom_x_lowvol_20_10', # IC=6.84%, IR=0.17
        ],
        'direction': {
            'fund_profit_growth': 1,
            'mom_x_lowvol_20_20': 1,
            'mom_x_lowvol_20_10': 1,
        },
        'weights': [0.45, 0.30, 0.25],
    },

    # === 消费/传媒/农业/环保/医药 ===
    '消费/传媒/农业/环保/医药': {
        'factors': [
            'fund_eps',     # IC=5.46%, IR=0.18
            'fund_score',   # IC=5.22%, IR=0.18
            'fund_roe',    # IC=4.24%, IR=0.15
            'bb_width',    # IC=5.45%, IR=0.19
        ],
        'direction': {
            'fund_score': 1,
            'fund_eps': 1,
            'fund_roe': 1,
            'bb_width': 1,
        },
        'weights': [0.30, 0.28, 0.22, 0.20],
    },

    # === 金融 ===
    '金融': {
        'factors': [
            'fund_score',          # IC=7.36%, IR=0.20
            'fund_roe',            # IC=6.35%, IR=0.17
            'fund_profit_growth',  # IC=5.67%, IR=0.16
            'fund_revenue_growth', # IC=5.48%, IR=0.18
            'ret_vol_ratio_10',   # IC=4.78%, IR=0.13
            'rsi_14',             # IC=4.71%, IR=0.12 (RSI高=超买，要反向)
        ],
        'direction': {
            'fund_score': 1,
            'fund_roe': 1,
            'fund_profit_growth': 1,
            'fund_revenue_growth': 1,
            'ret_vol_ratio_10': 1,
            'rsi_14': -1,  # RSI高=超买，应卖出
        },
        'weights': [0.25, 0.20, 0.18, 0.15, 0.12, 0.10],
    },

    # === 交运 ===
    '交运': {
        'factors': [
            'mom_x_lowvol_20_20', # IC=6.19%, IR=0.16
            'mom_x_lowvol_20_10', # IC=5.96%, IR=0.16
            'fund_score',         # IC=5.02%, IR=0.15
        ],
        'direction': {
            'mom_x_lowvol_20_20': 1,
            'mom_x_lowvol_20_10': 1,
            'fund_score': 1,
        },
        'weights': [0.40, 0.35, 0.25],
    },

    # === 化工 ===
    '化工': {
        'factors': [
            'fund_profit_growth', # IC=5.04%, IR=0.21
            'fund_cf_to_profit',  # IC=3.30%, IR=0.16
            'mom_x_lowvol_10_20',# IC=3.00%, IR=0.13
            'mom_x_lowvol_10_10',# IC=2.95%, IR=0.13
        ],
        'direction': {
            'fund_profit_growth': 1,
            'fund_cf_to_profit': 1,
            'mom_x_lowvol_10_20': 1,
            'mom_x_lowvol_10_10': 1,
        },
        'weights': [0.40, 0.25, 0.20, 0.15],
    },
}


# 行业分类到具体行业的映射
# 用于将大类行业映射到具体的行业配置
INDUSTRY_CATEGORY_MAPPING = {
    '科技/成长': ['互联网/软件', '通信/计算机', '半导体/光伏', '电子'],
    '周期/资源': ['有色/钢铁/煤炭/建材', '化工', '基建/地产/石油石化'],
    '消费/稳定': ['消费/传媒/农业/环保/医药'],
    '金融/大盘': ['金融'],
    '新能源': ['电力设备', '新能源车/风电', '半导体/光伏'],
    '制造业': ['自动化/制造', '军工', '交运'],
}


def get_industry_config(industry_name: str) -> dict:
    """
    获取指定行业的因子配置

    Args:
        industry_name: 行业名称

    Returns:
        行业因子配置字典，如果没有配置则返回None
    """
    return INDUSTRY_FACTOR_CONFIG.get(industry_name)


def get_factor_direction(industry_name: str, factor_name: str) -> int:
    """
    获取指定行业中因子的方向

    Args:
        industry_name: 行业名称
        factor_name: 因子名称

    Returns:
        1=正向(因子值高买), -1=反向(因子值低买), 0=未知
    """
    config = get_industry_config(industry_name)
    if config:
        return config.get('direction', {}).get(factor_name, 0)
    return 0


def get_industry_factors(industry_name: str) -> list:
    """
    获取指定行业的因子列表

    Args:
        industry_name: 行业名称

    Returns:
        因子名称列表
    """
    config = get_industry_config(industry_name)
    if config:
        return config.get('factors', [])
    return []


def get_industry_weights(industry_name: str) -> list:
    """
    获取指定行业的因子权重

    Args:
        industry_name: 行业名称

    Returns:
        权重列表
    """
    config = get_industry_config(industry_name)
    if config:
        return config.get('weights', [])
    return []
