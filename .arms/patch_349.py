#!/usr/bin/env python3
"""#349 矩阵层宇宙修复补丁器 — 同一份编辑施加于 iso 副本与生产 (无转写漂移).

用法: python patch_349.py <target.py>
- 循环内: 装载时用 stock_pool._is_non_stock_series (Fix#44 同源 VWAP 恒等式)
  判别非个股序列(裸代码指数/海外股), 跳过装载并记名.
- 循环后: 整列剔除 + 重建 code_to_idx/idx_to_code.
幂等: 已含 '#349' 标记则跳过.
"""
import sys

target = sys.argv[1]
s = open(target, encoding='utf-8').read()
if '#349' in s:
    print('已含 #349 标记, 跳过')
    sys.exit(0)

old1 = """        df = df.set_index('datetime')
        df = df[df.index >= pd.Timestamp(fromdate)]
        df = df[df.index <= pd.Timestamp(todate)]
        df = df.reindex(calendar)"""
new1 = """        df = df.set_index('datetime')
        df = df[df.index >= pd.Timestamp(fromdate)]
        df = df[df.index <= pd.Timestamp(todate)]
        # #349 (2026-10-23): 矩阵宇宙此前只按文件名排 sh000001 — sh000852/399006 +
        # 27个裸代码指数/海外股仍以死列进矩阵 (实测 signals/selections/trade_realized
        # ∩ 非个股 = 0; 池层 Fix#44 已拦). 装载时用 stock_pool._is_non_stock_series
        # 同源 VWAP 恒等式判别, 整列剔除. 恒等证明 = iso 臂逐位复现 anchor.
        if _is_non_stock_series(df):
            _nonstock.append(code)
            continue
        df = df.reindex(calendar)"""
assert s.count(old1) == 1, f'anchor1 x{s.count(old1)}'
s = s.replace(old1, new1)

old2 = """    for j, code in enumerate(tqdm(stock_codes, desc="loading")):"""
new2 = """    from core.stock_pool import _is_non_stock_series
    _nonstock = []  # #349: 非个股序列(指数/海外股)死列清单
    for j, code in enumerate(tqdm(stock_codes, desc="loading")):"""
assert s.count(old2) == 1, f'anchor2 x{s.count(old2)}'
s = s.replace(old2, new2)

old3 = """    del df

    # 预计算滚动ADV矩阵"""
new3 = """    del df
    if _nonstock:  # #349: 死列剔除 — 逐列矩阵无跨列消费, 索引重编号自洽
        _keep = [j for j, c in enumerate(stock_codes) if c not in set(_nonstock)]
        stock_codes = [stock_codes[j] for j in _keep]
        close_px, open_px, low_px = close_px[:, _keep], open_px[:, _keep], low_px[:, _keep]
        volume_m, amount_m = volume_m[:, _keep], amount_m[:, _keep]
        code_to_idx = {c: i for i, c in enumerate(stock_codes)}
        idx_to_code = {i: c for i, c in enumerate(stock_codes)}
        print(f"矩阵宇宙: 剔除非个股序列 {len(_nonstock)} 个 (Fix#44 同源判别, #349) "
              f"-> {len(stock_codes)} 列")

    # 预计算滚动ADV矩阵"""
assert s.count(old3) == 1, f'anchor3 x{s.count(old3)}'
s = s.replace(old3, new3)

open(target, 'w', encoding='utf-8').write(s)
print(f'#349 补丁已施加: {target}')
