"""
为所有行业添加 bear_factors/bear_ic 到 factor_config.yaml
基于 find_bear_factors.py 的实证结果
"""
import yaml
import os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 实证TOP3熊市因子 (跨行业IC稳定>0)
# V1因子(V1把2022从-29%提到-9.6%), 配合趋势门控保护牛市
BEAR_FACTORS = ['trend_vol', 'volatility', 'inv_turnover']
BEAR_WEIGHTS = [0.40, 0.35, 0.25]
BEAR_IC = 0.065
BEAR_COMBINED_IC = 0.085

config_path = os.path.join(BASE, 'config', 'factor_config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 统计
added = 0
skipped = 0
industry_factors = config.get('industry_factors', {})
for industry, cfg in industry_factors.items():
    if not isinstance(cfg, dict):
        skipped += 1
        continue
    if 'bear_factors' in cfg:
        # 覆盖旧值
        pass

    cfg['bear_factors'] = BEAR_FACTORS
    cfg['bear_weights'] = BEAR_WEIGHTS
    cfg['bear_ic'] = BEAR_IC
    cfg['bear_combined_ic'] = BEAR_COMBINED_IC
    added += 1

# 保存 (保留格式)
with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
print(f"已添加bear_factors: {added}个行业, 跳过: {skipped}个 (已有bear_factors或非dict)")
print(f"因子: {BEAR_FACTORS}")
print(f"IC: bear_ic={BEAR_IC}, bear_combined_ic={BEAR_COMBINED_IC}")
