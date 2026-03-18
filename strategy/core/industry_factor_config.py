# core/industry_factor_config.py
"""
行业因子配置模块 - 基于真滚动验证结果

更新记录:
- 2026-03-18: 基于 true_rolling_validator.py 结果更新
  - 使用Top3因子等权，更稳健
  - 基于真滚动验证中各行业稳定有效的因子

核心发现:
1. 基本面因子在A股最稳健: fund_profit_growth, fund_score, fund_roe
2. 动量×低波动因子次之: mom_x_lowvol_20_20, mom_x_lowvol_20_10
3. 避免过度定制化，使用Top3等权降低过拟合
"""

# 基于真滚动验证中出现频率最高的因子
# 这些因子在多个时点被反复选中，说明具有时间稳定性
STABLE_FACTORS = {
    'fundamental': [
        'fund_profit_growth',   # 出现228次，最稳定
        'fund_score',           # 出现163次
        'fund_cf_to_profit',    # 出现134次
        'fund_roe',             # 出现125次
        'fund_revenue_growth',  # 出现125次
        'fund_gross_margin',    # 出现102次
    ],
    'technical': [
        'mom_x_lowvol_20_20',   # 出现126次
        'mom_x_lowvol_20_10',   # 出现48次
        'bb_rsi_combo',         # 出现27次
        'rsi_vol_combo',        # 出现45次
    ]
}


# 行业因子配置 - 基于真滚动验证结果
# 策略: Top3因子等权，减少过拟合
INDUSTRY_FACTOR_CONFIG = {
    # === 自动化/制造 === IR=0.38 (真滚动最高)
    '自动化/制造': {
        'factors': ['fund_revenue_growth', 'mom_x_lowvol_20_20', 'mom_x_lowvol_20_10'],
        'method': 'equal',
    },

    # === 电力设备 === IR=0.36
    '电力设备': {
        'factors': ['fund_score', 'fund_profit_growth', 'mom_x_lowvol_20_20'],
        'method': 'equal',
    },

    # === 基建/地产/石油石化 === IR=0.34
    '基建/地产/石油石化': {
        'factors': ['fund_profit_growth', 'fund_score', 'fund_roe'],
        'method': 'equal',
    },

    # === 金融 === IR=0.30
    '金融': {
        'factors': ['fund_cf_to_profit', 'fund_score', 'fund_roe'],
        'method': 'equal',
    },

    # === 化工 === IR=0.28
    '化工': {
        'factors': ['mom_x_lowvol_20_20', 'mom_x_lowvol_20_10', 'fund_gross_margin'],
        'method': 'equal',
    },

    # === 新能源车/风电 === IR=0.27
    '新能源车/风电': {
        'factors': ['fund_profit_growth', 'fund_revenue_growth', 'fund_score'],
        'method': 'equal',
    },

    # === 有色/钢铁/煤炭/建材 === IR=0.26
    '有色/钢铁/煤炭/建材': {
        'factors': ['fund_score', 'fund_revenue_growth', 'fund_profit_growth'],
        'method': 'equal',
    },

    # === 军工 === IR=0.24
    '军工': {
        'factors': ['mom_x_lowvol_20_20', 'mom_x_lowvol_20_10', 'fund_profit_growth'],
        'method': 'equal',
    },

    # === 互联网/软件 === IR=0.19
    '互联网/软件': {
        'factors': ['mom_x_lowvol_20_20', 'mom_x_lowvol_20_10', 'fund_roe'],
        'method': 'equal',
    },

    # === 电子 === IR=0.18
    '电子': {
        'factors': ['mom_x_lowvol_20_20', 'fund_profit_growth', 'fund_revenue_growth'],
        'method': 'equal',
    },

    # === 消费/传媒/农业/环保/医药 === IR=0.17
    '消费/传媒/农业/环保/医药': {
        'factors': ['fund_revenue_growth', 'fund_profit_growth', 'mom_x_lowvol_20_20'],
        'method': 'equal',
    },

    # === 半导体/光伏 === IR=0.12
    '半导体/光伏': {
        'factors': ['mom_x_lowvol_20_20', 'fund_revenue_growth', 'fund_profit_growth'],
        'method': 'equal',
    },

    # === 交运 === IR=0.05 (效果差，标记为低置信度)
    '交运': {
        'factors': ['fund_profit_growth', 'fund_gross_margin', 'mom_x_lowvol_20_20'],
        'method': 'equal',
        'confidence': 'low',  # 信号置信度低，建议减配或不做
    },

    # === 通信/计算机 === IR=-0.04 (效果差，标记为低置信度)
    '通信/计算机': {
        'factors': ['fund_profit_growth', 'fund_score', 'fund_revenue_growth'],
        'method': 'equal',
        'confidence': 'low',  # 信号置信度低，建议减配或不做
    },
}

# 低置信度行业列表（IR < 0.1 或 IC为负）
LOW_CONFIDENCE_INDUSTRIES = ['交运', '通信/计算机']

# 默认因子配置（用于无法识别行业的股票）
DEFAULT_FACTORS = {
    'factors': ['fund_profit_growth', 'fund_score', 'mom_x_lowvol_20_20'],
    'method': 'equal',
}


def get_industry_config(industry_name: str) -> dict:
    """获取行业因子配置"""
    return INDUSTRY_FACTOR_CONFIG.get(industry_name, DEFAULT_FACTORS)


def get_industry_factors(industry_name: str) -> list:
    """获取行业因子列表"""
    config = get_industry_config(industry_name)
    return config.get('factors', DEFAULT_FACTORS['factors'])


def get_stable_factors(factor_type: str = None) -> list:
    """获取稳定因子列表"""
    if factor_type:
        return STABLE_FACTORS.get(factor_type, [])
    return STABLE_FACTORS['fundamental'] + STABLE_FACTORS['technical']
