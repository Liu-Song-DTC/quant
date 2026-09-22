#!/usr/bin/env python
"""实盘成交成本反馈回路 (fill_cost_feedback, 2026-09-22)

成本审计(9/21-22)裁决的落地件: "实盘恢复后用真实成交价回填, 下次自然重锚时
校准slip_rate"。本工具把真实QMT成交流水与回测成本模型逐笔对账:

  模型假设(bt_execution.py:41-46 + yaml):
    滑点  万10双边 (open×(1±0.001))
    佣金  万5双边 (真实≈万1-2.5 → 模型多收 +3.2~5.2% NAV, 中心+3.9%@万2)
    印花  万5卖出单边 全程 (真实=万10 → 2023-08-28 万5 → 模型历史少收 −4.4%)
    过户  无 (真实万0.1双边, 可忽略)

  实盘路径: 收盘后出单(trade_orders.json) → 次日开盘附近成交。
  因此逐笔基准 = 成交当日开盘价(raw_data/{code}/none.csv, 未复权真实价),
  真实滑点(buy) = fill_px/open − 1, (sell) = (open − fill_px)/open。
  每笔差额 = qty×px×(模型费率 − 真实费率), 正=实盘优于模型。

输出:
  1) 逐笔表(code/date/侧/股数/成交价/开盘基准/滑点bp)
  2) 按侧聚合: n/中位/均值/p10/p90 滑点
  3) 四项成本差额合计(元) + 占总成交额bp
  4) slip_rate校准建议 = 各侧p90×1.5缓冲(下限万3, 双侧样本≥5时拆分买/卖) +
     滑点响应面NAV映射 (万10→万5=+6.12pp双边已探; 不对称分解9/22: 双侧弹性对称
     1.01+线性可加 → 分侧独立校准零交互惩罚, 每侧每降万5≈+3.11pp)
  5) 报告md落盘(--out, 默认 /tmp/fill_cost_report.md)

用法:
  python strategy/tools/fill_cost_feedback.py --fills 成交.csv
    [--commission-bp 2.0] [--raw-root data/stock_data/raw_data] [--out report.md]

fills CSV列名容错(QMT中文导出/英文均可): 日期(成交日期/日期/date),
代码(证券代码/代码/code), 方向(买卖标志/买卖方向/方向/side, 买|B|1=买入,
卖|S|2=卖出), 数量(成交数量/数量/qty/shares), 价格(成交均价/成交价格/价格/price)。
编码自动utf-8-sig/gbk。跳过标题垃圾行(按核心列解析失败逐行报告)。

独立零耦合: 不import任何strategy模块, 不在信号指纹清单内, 只读数据文件。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parents[2]

# 回测成本模型常量 (bt_execution.py:41-46 + yaml, 与生产一致)
MODEL_SLIP = 0.001       # 万10双边
MODEL_COMM = 0.0005      # 万5双边
MODEL_STAMP = 0.0005     # 万5卖出单边(模型全程flat)
TRANSFER = 0.00001       # 真实过户费万0.1双边(模型无此项)
STAMP_CUT_DATE = pd.Timestamp('2023-08-28')  # 真实印花税万10→万5时点

# 滑点响应面 (9/22 probe_slippage_bracket, 严格单调确定性成本函数):
#   万5 → 777,549 (+6.12pp vs 生产732,689)   万15 → 670,298 (−8.52pp)
#   下行(降成本)为算术区(每万5步+6.1pp), 上行放大~1.3×。
SLIP_SURFACE_PP_PER_5BP = 6.12  # 万5→万10之间每降万5双边滑点 ≈ +6.12pp NAV
SLIP_SURFACE_PROBED_MIN_BP = 5  # 已探最低臂=万5; 更低为算术区线性外推

_SIDE_ALIASES_BUY = ('买', 'b', '1')
_SIDE_ALIASES_SELL = ('卖', 's', '2')


def _pick_col(df, candidates):
    for c in df.columns:
        s = str(c).strip().lower().replace(' ', '')
        for cand in candidates:
            if s == cand or (len(cand) >= 2 and cand in s):
                return c
    return None


def _read_fills(path):
    p = Path(path)
    if not p.exists():
        print(f"✗ fills文件不存在: {p}")
        sys.exit(1)
    for enc in ('utf-8-sig', 'gbk', 'utf-8'):
        try:
            df = pd.read_csv(p, encoding=enc, dtype=str)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        print("✗ fills文件编码无法解析 (尝试过 utf-8-sig/gbk/utf-8)")
        sys.exit(1)

    col_date = _pick_col(df, ('date', '日期', '成交日期', '交易日期', 'trade_date'))
    col_code = _pick_col(df, ('code', '代码', '证券代码', '股票代码', 'stock_code'))
    col_side = _pick_col(df, ('side', '方向', '买卖', '委托方向', 'bs'))
    col_qty = _pick_col(df, ('qty', 'shares', 'vol', '数量', '成交数量', '成交量'))
    col_px = _pick_col(df, ('price', 'fill', 'avg', '价格', '均价', '成交均价', '成交价格', '成交价'))
    missing = [n for n, c in (('日期', col_date), ('代码', col_code), ('方向', col_side),
                              ('数量', col_qty), ('价格', col_px)) if c is None]
    if missing:
        print(f"✗ fills列名无法识别, 缺: {missing}. 实际列: {list(df.columns)}")
        sys.exit(1)

    out = pd.DataFrame({
        'date': pd.to_datetime(df[col_date], errors='coerce'),
        'code': (df[col_code].astype(str).str.extract(r'(\d{6})')[0]),
        'side_raw': df[col_side].astype(str).str.strip().str.lower(),
        'qty': pd.to_numeric(df[col_qty], errors='coerce'),
        'price': pd.to_numeric(df[col_px], errors='coerce'),
    })
    def _side_of(v):
        if v.startswith(_SIDE_ALIASES_BUY):
            return 'buy'
        if v.startswith(_SIDE_ALIASES_SELL):
            return 'sell'
        return None
    out['side'] = out['side_raw'].map(_side_of)
    bad = out[out[['date', 'code', 'side', 'qty', 'price']].isna().any(axis=1)]
    out = out.dropna(subset=['date', 'code', 'side', 'qty', 'price'])
    print(f"fills解析: {len(out)}行有效, {len(bad)}行剔除(核心列缺失/无法解析)")
    if len(bad) and len(bad) <= 10:
        print(bad.head(10).to_string(index=False))
    return out


def _load_open(code, date, raw_root):
    """成交当日开盘价: raw_data/{code}/none.csv (未复权真实交易价)."""
    none_f = Path(raw_root) / code / 'none.csv'
    qfq_f = _PROJECT_DIR / 'data' / 'stock_data' / 'backtrader_data' / f'{code}_qfq.csv'
    f, src = (none_f, 'none') if none_f.exists() else (qfq_f, 'qfq(警告:复权价可能≠真实成交价)')
    if not f.exists():
        return np.nan, f'no-file:{code}'
    try:
        df = pd.read_csv(f, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(f, encoding='gbk')
    dcol = '日期' if '日期' in df.columns else 'datetime'
    ocol = '开盘' if '开盘' in df.columns else 'open'
    df[dcol] = pd.to_datetime(df[dcol])
    row = df[df[dcol] == pd.Timestamp(date)]
    if row.empty:
        return np.nan, f'no-bar:{code}@{date.date()}'
    o = float(row.iloc[0][ocol])
    return (o if o > 0 else np.nan), src


def _real_stamp(date, autodate):
    if not autodate:
        return MODEL_STAMP
    return 0.001 if pd.Timestamp(date) < STAMP_CUT_DATE else 0.0005


def _banner(t):
    print(f"\n===== {t} =====")


def main():
    ap = argparse.ArgumentParser(description='实盘成交成本反馈回路: 真实成交 vs 回测成本模型')
    ap.add_argument('--fills', required=True, help='QMT成交流水CSV (列名容错, utf-8-sig/gbk)')
    ap.add_argument('--commission-bp', type=float, default=2.0, help='真实佣金率(万), 默认2.0')
    ap.add_argument('--no-stamp-autodate', action='store_true',
                    help='印花税不按2023-08-28分档, 全程按万5')
    ap.add_argument('--raw-root', default=str(_PROJECT_DIR / 'data' / 'stock_data' / 'raw_data'),
                    help='未复权raw数据根目录')
    ap.add_argument('--out', default='/tmp/fill_cost_report.md', help='报告输出路径')
    args = ap.parse_args()

    real_comm = args.commission_bp / 10000.0
    fills = _read_fills(args.fills)
    if fills.empty:
        print("✗ 无有效成交行, 中止")
        sys.exit(1)

    _banner('逐笔基准对齐 (open=成交当日开盘, raw none.csv)')
    opens, srcs = [], []
    for _, r in fills.iterrows():
        o, s = _load_open(r['code'], r['date'], args.raw_root)
        opens.append(o)
        srcs.append(s)
    fills['open'] = opens
    src_bad = fills[[s.startswith('qfq') for s in srcs]]
    no_bar = fills[fills['open'].isna()]
    if len(src_bad):
        print(f"⚠ {len(src_bad)}笔基准来自qfq(复权价)而非none.csv — 近期有除权除息则偏差")
    if len(no_bar):
        print(f"⚠ {len(no_bar)}笔当日无K线(停牌/数据缺失), 滑点统计剔除, 成本项按模型滑点计")

    fills['turnover'] = fills['qty'] * fills['price']
    fills['slip'] = np.where(fills['side'] == 'buy',
                             fills['price'] / fills['open'] - 1,
                             (fills['open'] - fills['price']) / fills['open'])
    fills['slip_bp'] = fills['slip'] * 10000
    fills['real_stamp'] = fills.apply(lambda r: _real_stamp(r['date'], not args.no_stamp_autodate)
                                      if r['side'] == 'sell' else 0.0, axis=1)
    # 每笔差额(元): 模型收取 − 真实支付, 正=实盘优于模型
    fills['gap_comm'] = fills['turnover'] * (MODEL_COMM - real_comm)
    fills['gap_stamp'] = np.where(fills['side'] == 'sell',
                                  fills['turnover'] * (MODEL_STAMP - fills['real_stamp']), 0.0)
    fills['gap_slip'] = fills['turnover'] * (MODEL_SLIP - fills['slip'].fillna(MODEL_SLIP))
    fills['gap_transfer'] = -fills['turnover'] * TRANSFER

    _banner('逐笔明细 (前20)')
    show = fills[['date', 'code', 'side', 'qty', 'price', 'open', 'slip_bp',
                  'gap_comm', 'gap_stamp', 'gap_slip']].head(20)
    print(show.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))

    _banner('滑点按侧聚合 (bp, 正=劣于开盘)')
    ok = fills[fills['open'].notna()]
    for side in ('buy', 'sell'):
        s = ok[ok['side'] == side]['slip_bp']
        if s.empty:
            print(f"  {side}: 无样本")
            continue
        print(f"  {side}: n={len(s)}  中位{s.median():+.1f}  均值{s.mean():+.1f}  "
              f"p10={s.quantile(0.10):+.1f}  p90={s.quantile(0.90):+.1f}  "
              f"range=[{s.min():+.1f}, {s.max():+.1f}]")

    _banner('成本项差额合计 (正=实盘优于模型)')
    tot_turnover = fills['turnover'].sum()
    rows = [('佣金(模型万5 vs 真实万{:.0f})'.format(args.commission_bp), fills['gap_comm'].sum()),
            ('印花税(模型万5 flat vs 真实分档)', fills['gap_stamp'].sum()),
            ('滑点(模型万10 vs 真实成交)', fills['gap_slip'].sum()),
            ('过户费(模型无 vs 真实万0.1)', fills['gap_transfer'].sum())]
    for name, v in rows:
        print(f"  {name}: {v:+,.2f} 元 ({v / tot_turnover * 10000:+.1f}bp)")
    tot = sum(v for _, v in rows)
    print(f"  合计: {tot:+,.2f} 元 ({tot / tot_turnover * 10000:+.1f}bp 占总成交额)")

    _banner('slip_rate校准建议 (下次自然重锚时改yaml slippage)')
    rec = None
    if ok.empty:
        print("  无有效滑点样本 — 保持生产万10不动, 待成交积累后重跑本工具")
    else:
        buy_s = ok[ok['side'] == 'buy']['slip_bp']
        sell_s = ok[ok['side'] == 'sell']['slip_bp']
        if min(len(buy_s), len(sell_s)) >= 5:
            # 分侧建议 (9/22不对称分解探针: 双边弹性对称1.01+线性可加+0.07%
            # → 分侧校准=纯增益零交互惩罚, 各侧按自己的p90×1.5缓冲)
            p90b = max(buy_s.quantile(0.90), 0.0)
            p90s = max(sell_s.quantile(0.90), 0.0)
            rec_buy = max(3.0, np.ceil(np.round(p90b * 1.5, 6)))
            rec_sell = max(3.0, np.ceil(np.round(p90s * 1.5, 6)))
            rec = (rec_buy + rec_sell) / 2 / 10000.0  # 报告档用平均
            print(f"  真实滑点: 买侧 n={len(buy_s)} 中位{buy_s.median():+.1f} p90={p90b:+.1f}bp | "
                  f"卖侧 n={len(sell_s)} 中位{sell_s.median():+.1f} p90={p90s:+.1f}bp")
            print(f"  建议: 买侧万{rec_buy:.0f} / 卖侧万{rec_sell:.0f} (各侧p90×1.5缓冲, 下限万3)")
            if rec_buy != rec_sell:
                print("  两侧不等 → yaml需拆 slippage_buy/slippage_sell 双键(代码改动, "
                      "自然重锚窗口顺带); 不对称探针证实分侧独立校准无交互惩罚")
            for side, rc, p90x in (('买侧', rec_buy, p90b), ('卖侧', rec_sell, p90s)):
                if rc > 10:
                    print(f"  ⚠ {side}真实p90={p90x:.1f}bp高于模型万10 — 上调至万{rc:.0f}或优化执行")
            d_b = 10 - rec_buy
            d_s = 10 - rec_sell
            if d_b + d_s > 0:
                # 线性平面: 每侧每降万5 ≈ +3.11pp NAV (arm2/arm3: +22,806/+23,066元)
                uplift = 3.11 * d_b / 5 + 3.11 * d_s / 5
                print(f"  滑点响应面映射(线性平面): 万10→(万{rec_buy:.0f},万{rec_sell:.0f}) "
                      f"≈ +{uplift:.1f}pp NAV (分侧臂+3.11pp/万5为已探锚点)")
            else:
                print("  建议与生产一致 — 无需改")
        else:
            p90 = max(ok['slip_bp'].quantile(0.90), 0.0)
            med = ok['slip_bp'].median()
            # 建议 = 真实p90×1.5缓冲, 下限万3 (yaml接受任意浮点, 无需万取整)
            rec_bp = max(3.0, np.ceil(np.round(p90 * 1.5, 6)))
            rec = rec_bp / 10000.0
            print(f"  真实滑点: 中位{med:+.1f}bp, p90={p90:+.1f}bp (双边合并{len(ok)}笔, "
                  f"样本不足不拆分)")
            if rec_bp > 10:
                print(f"  ⚠ 真实p90高于模型万10 — 建议上调至万{rec_bp:.0f}或优化执行"
                      f"(限价单开盘内成交), 重锚时一并改")
            else:
                print(f"  建议 slip_rate = 万{rec_bp:.0f} (p90×1.5缓冲, 下限万3)")
                delta_bp = 10 - rec_bp
                if delta_bp > 0:
                    uplift = SLIP_SURFACE_PP_PER_5BP * delta_bp / 5
                    qual = '外推' if rec_bp < SLIP_SURFACE_PROBED_MIN_BP else '已探区'
                    print(f"  滑点响应面映射: 万10→万{rec_bp:.0f} ≈ +{uplift:.1f}pp NAV "
                          f"({qual}: 万5臂+6.12pp为已探锚点, 响应面严格单调无膝点)")
                else:
                    print("  建议与生产一致 — 无需改")
    # 佣金/印花是factual项: 模型万5佣金 vs 真实万2 = 确定性多收, 直接报年化意义
    comm_bp = fills['gap_comm'].sum() / tot_turnover * 10000
    print(f"  佣金factual项: 模型每笔多收{comm_bp:+.1f}bp — 重锚时可顺带改yaml commission"
          f"(与印花税修复同一窗口, 零额外代价)")

    _banner(f'报告落盘: {args.out}')
    lines = ['# 实盘成交成本反馈报告',
             f'\n生成: {pd.Timestamp.now():%Y-%m-%d %H:%M}  样本: {len(fills)}笔  '
             f'总额{tot_turnover:,.0f}元',
             '\n## 逐笔滑点 (bp, 正=劣于开盘基准)', '']
    lines.append(fills[['date', 'code', 'side', 'qty', 'price', 'open', 'slip_bp']]
                 .to_markdown(index=False))
    lines += ['\n## 成本差额 (正=实盘优于模型)', '',
              f'佣金 {fills["gap_comm"].sum():+,.2f}元, 印花 {fills["gap_stamp"].sum():+,.2f}元, '
              f'滑点 {fills["gap_slip"].sum():+,.2f}元, 过户 {fills["gap_transfer"].sum():+,.2f}元',
              f'合计 {tot:+,.2f}元']
    if rec is not None:
        _p90x = f'{p90:.1f}' if 'p90' in dir() else '—'
        if 'rec_buy' in dir():
            _rec_txt = (f'yaml slippage 0.001 → 买侧{rec_buy/10000:.4f}/卖侧{rec_sell/10000:.4f} '
                        f'(各侧p90×1.5缓冲; 不对称探针9/22证实分侧独立校准无交互惩罚)')
        else:
            _rec_txt = (f'yaml slippage 0.001 → {rec} (p90={_p90x}bp×1.5缓冲, 下限万3)'
                        if rec_bp <= 10 else
                        f'⚠ 真实p90={_p90x}bp高于模型万10 — 建议上调至{rec}或优化执行')
        lines += ['\n## 校准建议',
                  f'下次自然重锚: {_rec_txt}, '
                  f'commission 0.0005 → {real_comm} (factual修复, 同一窗口顺带)']
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print('完成')


if __name__ == '__main__':
    main()
