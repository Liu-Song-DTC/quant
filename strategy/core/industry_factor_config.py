# core/industry_factor_config.py
"""
行业因子配置模块 - 基于分行业因子验证结果

从 factor_validation_results/all_results.csv 中提取各行业Top因子
配置结构: {行业名: {factors: [因子列表], method: 'top1'|'weighted'}}
"""

INDUSTRY_FACTOR_CONFIG = {
    # === 电力设备 === Top1
    '电力设备': {
        'factors': ['fund_score'],
        'method': 'top1',
    },

    # === 新能源车/风电 === Top1
    '新能源车/风电': {
        'factors': ['fund_profit_growth'],
        'method': 'top1',
    },

    # === 互联网/软件 === Top1
    '互联网/软件': {
        'factors': ['mom_x_lowvol_20_10'],
        'method': 'top1',
    },

    # === 基建/地产/石油石化 === 加权组合（有效）
    '基建/地产/石油石化': {
        'factors': [
            'fund_profit_growth', # IC=7.24%
            'fund_roe',           # IC=6.86%
            'fund_score',         # IC=5.62%
            'fund_eps',           # IC=4.26%
        ],
        'weights': [0.30, 0.30, 0.25, 0.15],
        'method': 'weighted',
    },

    # === 有色/钢铁/煤炭/建材 === Top1
    '有色/钢铁/煤炭/建材': {
        'factors': ['fund_score'],
        'method': 'top1',
    },

    # === 自动化/制造 === Top1
    '自动化/制造': {
        'factors': ['fund_score'],
        'method': 'top1',
    },

    # === 电子 === Top1
    '电子': {
        'factors': ['fund_profit_growth'],
        'method': 'top1',
    },

    # === 半导体/光伏 === Top1
    '半导体/光伏': {
        'factors': ['mom_x_lowvol_20_20'],
        'method': 'top1',
    },

    # === 通信/计算机 === Top1
    '通信/计算机': {
        'factors': ['fund_score'],
        'method': 'top1',
    },

    # === 军工 === Top1
    '军工': {
        'factors': ['fund_profit_growth'],
        'method': 'top1',
    },

    # === 消费/传媒/农业/环保/医药 === Top1
    '消费/传媒/农业/环保/医药': {
        'factors': ['fund_eps'],
        'method': 'top1',
    },

    # === 金融 === Top1
    '金融': {
        'factors': ['fund_score'],
        'method': 'top1',
    },

    # === 交运 === Top1
    '交运': {
        'factors': ['mom_x_lowvol_20_20'],
        'method': 'top1',
    },

    # === 化工 === Top1
    '化工': {
        'factors': ['fund_profit_growth'],
        'method': 'top1',
    },
}


def get_industry_config(industry_name: str) -> dict:
    return INDUSTRY_FACTOR_CONFIG.get(industry_name)


def get_industry_factors(industry_name: str) -> list:
    config = get_industry_config(industry_name)
    if config:
        return config.get('factors', [])
    return []
