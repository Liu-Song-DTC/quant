#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""N3探针 (2026-09-20): vol_control估计器变体敏感性 — 生产=组合自身20日std×√252,
阈值0.28 → clip(0.28/vol, 0.75, 1.0)×敞口 (portfolio.py:1034)。
变体: EWMA(λ=0.94) / EWMA(hl=20) / 半衰加权窗口(线性) / Parkinson(需HL, 跳过)。
一阶反事实: ΔNAV ≈ Σ r_t×Δm_t (m=敞口乘子), r=组合日收益(已含m_prod效应, 一阶近似)。
若变体对全期敞口-年的改变 <2% 或反事实ΔNAV <2% → N3关闭。
输入: 基线equity_curve (只读)。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQ = os.path.join(BASE, 'arms_20260919', 'C1_faithful_0_55', 'pre_equity_curve.csv')
TARGET = 0.28
LB = 20


def prod_vol(rets):
    """生产: 窗口20 std×√252, 逐日序列"""
    return rets.rolling(LB, min_periods=LB).std() * np.sqrt(252)


def ewma_vol(rets, hl):
    """EWMA vol: 半衰hl日, 初始化=前20日std"""
    alpha = 1 - np.exp(np.log(0.5) / hl)
    var = np.zeros(len(rets))
    init_std = prod_vol(rets).iloc[LB - 1]
    var[LB - 1] = init_std ** 2
    for i in range(LB, len(rets)):
        var[i] = (1 - alpha) * var[i - 1] + alpha * rets.iloc[i] ** 2
    return pd.Series(np.sqrt(var) * np.sqrt(252), index=rets.index)


def linw_vol(rets, hl=20):
    """线性递减权重窗口(std的加权版)"""
    w = np.arange(hl, 0, -1, dtype=float)
    w = w / w.sum()

    def _wstd(x):
        if len(x) < hl or np.isnan(x).any():
            return np.nan
        mu = np.sum(x * w)
        return np.sqrt(np.sum((x - mu) ** 2 * w))

    return rets.rolling(hl, min_periods=hl).apply(_wstd, raw=True) * np.sqrt(252)


def main():
    eq = pd.read_csv(EQ, parse_dates=['date'])
    r = eq['daily_ret'].dropna()
    print(f'日收益样本: {len(r)}')

    vols = {
        '生产(20d std)': prod_vol(r),
        # hl = ln(0.5)/ln(λ) → λ=0.94 ≈ hl 11.2
        'EWMA λ=0.94': ewma_vol(r, np.log(0.5) / np.log(0.94)),
        'EWMA hl=20': ewma_vol(r, 20),
        '线性加权20d': linw_vol(r, 20),
    }

    base = vols['生产(20d std)']
    bind = {}
    print('\n=== 各估计器: 超阈值(>0.28)天数 / 敞口乘子均值 / 与生产相关系数 ===')
    for name, v in vols.items():
        over = v > TARGET
        m = np.clip(TARGET / v, 0.75, 1.0)
        corr = v.corr(base)
        print(f"  {name}: 超阈值 {int(over.sum())}d ({over.mean()*100:.1f}%), "
              f"乘子均值 {m.mean():.4f} (生产 {np.clip(TARGET/base,0.75,1.0).mean():.4f}), "
              f"vol相关ρ={corr:.4f}")
        bind[name] = (v > TARGET)

    print('\n=== 绑定日集合重叠(生产超阈值日 vs 变体) ===')
    prod_bind = base > TARGET
    for name in ['EWMA λ=0.94', 'EWMA hl=20', '线性加权20d']:
        v = vols[name]
        vb = v > TARGET
        both = (prod_bind & vb).sum()
        only_prod = (prod_bind & ~vb).sum()
        only_var = (~prod_bind & vb).sum()
        print(f"  {name}: 同绑 {int(both)}  仅生产 {int(only_prod)}  仅变体 {int(only_var)}")

    print('\n=== 一阶反事实: ΔNAV ≈ Σ r_t×Δm_t ===')
    # m按估计器; 对生产m_prod与变体m_alt, ΔNAV/NAV ≈ Π(1+r_t×m_alt) − Π(1+r_t×m_prod)
    m_prod = np.clip(TARGET / base, 0.75, 1.0)
    rv = r.values
    for name in ['EWMA λ=0.94', 'EWMA hl=20', '线性加权20d']:
        v = vols[name]
        m_alt = np.clip(TARGET / v, 0.75, 1.0)
        valid = ~np.isnan(m_alt)
        nav_alt = np.prod(1 + rv[valid] * m_alt[valid])
        nav_prod = np.prod(1 + rv[valid] * m_prod[valid])
        print(f"  {name}: 反事实NAV {nav_alt:.4f} vs 生产 {nav_prod:.4f} "
              f"(Δ {(nav_alt-nav_prod)*100:+.2f}pp 全期)")

    print('\n=== 乘子<1的敞口-年损失(生产 vs 变体) ===')
    for name, v in vols.items():
        m = np.clip(TARGET / v, 0.75, 1.0)
        loss = (1 - m).sum() / 252
        print(f"  {name}: 敞口-年损失 {loss:.2f} (乘子<1累计/252)")

    print('\n=== 绑定日逐年分布(生产) ===')
    eq2 = eq.iloc[1:].reset_index(drop=True)  # 与dropna后的r对齐
    eq2['over'] = (base > TARGET).values
    eq2['yr'] = eq2['date'].dt.year
    for yr, g in eq2.groupby('yr'):
        print(f"  {yr}: 绑定日 {int(g['over'].sum())} / {len(g)}")


if __name__ == '__main__':
    main()
