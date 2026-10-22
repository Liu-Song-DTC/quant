#!/usr/bin/env python3
"""#348 2元闸口径修复补丁器 — VWAP → none.csv 真实收盘 (与实盘对齐), iso 与生产同源.

改动 (bt_execution.py):
  1. 模块级 RAW_DATA_PATH (= 实盘 generate_trade_orders.RAW_DATA_DIR 同源)
  2. 矩阵层新增 raw_close_m (none.csv 真实收盘, 窗口过滤, 无 ffill, 重复日期去重)
  3. 2元闸: raw_close 优先, 逐格缺失退回 VWAP (9d18716 兜底), 不退复权价
  4. #349 死列剔除块同步切 raw_close_m
  5. 覆盖度打印 (真实收盘格数/兜底格数)
幂等: 已含 '#348' 标记则跳过.
"""
import sys

target = sys.argv[1]
s = open(target, encoding='utf-8').read()
if '#348' in s:
    print('已含 #348 标记, 跳过')
    sys.exit(0)

# 1. 模块级 RAW_DATA_PATH (锚在 FUNDAMENTAL_PATH 行后)
old1 = "FUNDAMENTAL_PATH = config.get('paths.fundamental', os.path.join(_PROJECT_DIR, 'data/stock_data/fundamental_data/'))"
new1 = old1 + """
# #348 (2026-10-23): 2元闸真实收盘来源 — 与实盘 generate_trade_orders.RAW_DATA_DIR 同源
# (实盘 prices[code] < 2.0 读 none.csv 裸价; 回测此前用 VWAP, 口径不一致违反"实盘=回测").
RAW_DATA_PATH = os.path.join(_PROJECT_DIR, 'data/stock_data/raw_data')"""
assert s.count(old1) == 1, f'anchor1 x{s.count(old1)}'
s = s.replace(old1, new1)

# 2. raw_close_m 分配 (锚在 amount_m 分配行后)
old2 = "    amount_m = np.zeros_like(close_px)  # 真实成交额(元) — 文件自带列, 宇宙闸用(2026-10-17)"
new2 = old2 + """
    raw_close_m = np.full_like(close_px, np.nan)  # #348: none.csv 真实收盘, 2元闸用"""
assert s.count(old2) == 1, f'anchor2 x{s.count(old2)}'
s = s.replace(old2, new2)

# 3. 装载循环内读 raw (锚在 #349 非个股检查后, reindex 前)
old3 = """        if _is_non_stock_series(df):
            _nonstock.append(code)
            continue
        df = df.reindex(calendar)"""
new3 = """        if _is_non_stock_series(df):
            _nonstock.append(code)
            continue
        # #348: 读 none.csv 真实收盘 (窗口过滤, 重复日期去重, 无 ffill — 停牌日 NaN
        # ⇒ 该日不可交易, 与实盘"无价行⇒排除"一致). 失败/缺文件 ⇒ 全 NaN ⇒ 兜底 VWAP.
        _rp = os.path.join(RAW_DATA_PATH, code, 'none.csv')
        if os.path.exists(_rp):
            try:
                _rd = pd.read_csv(_rp, encoding='utf-8-sig')
                _rd.columns = [str(c).strip().lstrip('\\ufeff') for c in _rd.columns]
                if '日期' in _rd.columns and '收盘' in _rd.columns:
                    _rc = pd.Series(pd.to_numeric(_rd['收盘'], errors='coerce').values,
                                    index=pd.DatetimeIndex(
                                        pd.to_datetime(_rd['日期'], errors='coerce')))
                    _rc = _rc[_rc.index.notna()]
                    _rc = _rc[~_rc.index.duplicated()]
                    _rc = _rc[(_rc.index >= pd.Timestamp(fromdate)) &
                              (_rc.index <= pd.Timestamp(todate))]
                    raw_close_m[:, j] = _rc.reindex(calendar).values.astype(np.float32)
            except Exception as _e:
                print(f"  [raw收盘读取失败] {code}: {type(_e).__name__} {_e}")
        df = df.reindex(calendar)"""
assert s.count(old3) == 1, f'anchor3 x{s.count(old3)}'
s = s.replace(old3, new3)

# 4. #349 死列剔除块同步切 raw_close_m
old4 = "        volume_m, amount_m = volume_m[:, _keep], amount_m[:, _keep]"
new4 = old4 + "\n        raw_close_m = raw_close_m[:, _keep]"
assert s.count(old4) == 1, f'anchor4 x{s.count(old4)}'
s = s.replace(old4, new4)

# 5. 2元闸: raw_close 优先, 退回 VWAP
old5 = """    _raw_px = np.where((amount_m > 0) & (volume_m > 0),
                       amount_m / np.maximum(volume_m * 100.0, 1.0),
                       close_px).astype(np.float32)
    tradable = (~np.isnan(close_px)) & (_raw_px > 2.0) & (volume_m > 100) & (avg_daily_value > 5e6)"""
new5 = """    _vwap_px = np.where((amount_m > 0) & (volume_m > 0),
                        amount_m / np.maximum(volume_m * 100.0, 1.0),
                        close_px).astype(np.float32)
    # #348 (2026-10-23): 2元闸与实盘对齐 — 实盘读 none.csv 真实收盘
    # (generate_trade_orders.py:840); 回测此前用 VWAP, 全期翻转 2,157/6.88M 格
    # (0.03%, probe gate2_flip_1023), 锚点决策日交集 0. 真实收盘缺失逐格退回
    # VWAP 口径 (9d18716 兜底), 不退复权价.
    _raw_px = np.where(np.isnan(raw_close_m), _vwap_px, raw_close_m)
    _n_rc = int((~np.isnan(raw_close_m)).sum())
    print(f"2元闸口径(#348): 真实收盘 {_n_rc:,} 格 / 兜底VWAP {raw_close_m.size - _n_rc:,} 格")
    tradable = (~np.isnan(close_px)) & (_raw_px > 2.0) & (volume_m > 100) & (avg_daily_value > 5e6)"""
assert s.count(old5) == 1, f'anchor5 x{s.count(old5)}'
s = s.replace(old5, new5)

open(target, 'w', encoding='utf-8').write(s)
print(f'#348 补丁已施加: {target}')
