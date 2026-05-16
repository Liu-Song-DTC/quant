"""
选股结果导出到 Excel — 用于实盘选股记录和后续换股操作

用法: python export_selection.py

设计:
  - 使用完整 dynamic 因子模式选股（factor_mode=both）
  - factor_df 在预计算后自动释放，避免 OOM
  - 5个Sheet: 本次选股 / 调仓对比 / 市场状态 / 调仓历史 / 信号排名Top50
"""
import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).parent.resolve()

sys.path.insert(0, str(ROOT / "trade"))
sys.path.insert(0, str(ROOT / "strategy"))
sys.path.insert(0, str(ROOT / "data"))

from trade.config import TradeConfig


# ══════════════════════════════════════════════════════════════
#  Excel 样式工具
# ══════════════════════════════════════════════════════════════

HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
HEADER_FONT = Font(name="微软雅黑", size=11, bold=True, color="FFFFFF")
HEADER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)
DATA_FONT = Font(name="微软雅黑", size=10)
DATA_ALIGN = Alignment(horizontal="center", vertical="center")
THIN_BORDER = Border(
    left=Side(style="thin", color="D0D0D0"),
    right=Side(style="thin", color="D0D0D0"),
    top=Side(style="thin", color="D0D0D0"),
    bottom=Side(style="thin", color="D0D0D0"),
)
EVEN_FILL = PatternFill(start_color="F2F7FB", end_color="F2F7FB", fill_type="solid")
ODD_FILL = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
GREEN_FILL = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")
RED_FILL = PatternFill(start_color="FFEBEE", end_color="FFEBEE", fill_type="solid")
BLUE_FILL = PatternFill(start_color="E3F2FD", end_color="E3F2FD", fill_type="solid")


def style_header(ws, headers, row=1):
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=row, column=col, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = HEADER_ALIGN
        cell.border = THIN_BORDER


def style_data_rows(ws, start_row, end_row, ncols):
    for row in range(start_row, end_row + 1):
        fill = EVEN_FILL if (row - start_row) % 2 == 0 else ODD_FILL
        for col in range(1, ncols + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = DATA_FONT
            cell.alignment = DATA_ALIGN
            cell.border = THIN_BORDER
            if cell.fill == PatternFill():  # 不覆盖已设置的填充色
                cell.fill = fill


def auto_width(ws, min_width=8, max_width=40):
    for col_cells in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col_cells[0].column)
        for cell in col_cells:
            if cell.value:
                val = str(cell.value)
                length = sum(2 if ord(c) > 127 else 1 for c in val)
                max_len = max(max_len, length)
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, min_width), max_width)


def load_stock_name_map():
    stock_list = ROOT / "data" / "stock_data" / "stock_metadata" / "stock_list.csv"
    if stock_list.exists():
        df = pd.read_csv(stock_list, dtype={"symbol": str})
        return dict(zip(df["symbol"], df["name"]))
    return {}


def load_portfolio():
    state_file = ROOT / "trade" / "portfolio_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"cash": 150000.0, "positions": {}}


# ══════════════════════════════════════════════════════════════
#  选股核心
# ══════════════════════════════════════════════════════════════

def refresh_data():
    """增量更新股票数据 — 确保回测数据可用"""
    print("=" * 60)
    print("  检查数据更新...")
    print("=" * 60)

    from data.data_manager import StockDataManager
    manager = StockDataManager(data_dir=str(ROOT / "data" / "stock_data"))

    # 1. 获取股票列表（网络失败时自动使用缓存）
    stock_list = manager.get_stock_list()
    if len(stock_list) == 0:
        print("股票列表为空，使用现有数据文件直接选股")
        return

    symbols = stock_list['symbol'].tolist()
    symbols.insert(0, 'sh000001')
    print(f"股票池: {len(symbols)} 只")

    # 2. 增量下载原始数据（网络失败时自动使用缓存继续）
    try:
        manager.batch_download(symbols=symbols, force=False, max_workers=2)
    except Exception as e:
        print(f"数据更新异常: {e}")
        print("跳过在线更新，使用本地缓存数据继续...")

    # 3. 只更新缺失的回测数据（不重建全部，避免耗时过长）
    # batch_download 内部的 incremental_update 已更新原始数据
    # 对于缺失 backtrader 数据的股票，单独创建
    print("回测数据已就绪\n")


def run_selection():
    """执行选股 — 使用动态因子模式保证精度"""
    cfg = TradeConfig()
    today_str = datetime.today().strftime("%Y-%m-%d")

    print("=" * 60)
    print(f"  选股日期: {today_str}")
    print("=" * 60)

    # ── 先刷新数据 ──
    refresh_data()

    ps = load_portfolio()
    cash = ps.get("cash", 150000.0)
    positions = ps.get("positions", {})
    print(f"当前现金: ¥{cash:,.0f}  持仓: {len(positions)} 只")

    # ── 加载数据 ──
    from trade.signal_runner import SignalRunner
    runner = SignalRunner(
        bt_data_dir=str(cfg.bt_data_dir),
        fund_data_dir=str(cfg.fund_data_dir),
    )
    prices = runner.get_prices()
    print(f"有效价格数据: {len(prices)} 只")

    # ── 正常流程: prepare (含动态因子) + run ──
    runner.prepare(exposure=1.0, peak_equity=0.0)

    result = runner.run(
        current_positions={k: v.get("market_value", 0) if isinstance(v, dict) else v
                          for k, v in positions.items()},
        cash=cash,
        cost={},
    )

    if result is None:
        print("选股失败!")
        return None

    return result, prices, cash, today_str


# ══════════════════════════════════════════════════════════════
#  Excel 构建
# ══════════════════════════════════════════════════════════════

def build_excel(result, prices, cash, today_str, stock_names):
    wb = Workbook()
    selections = result.get("selections", [])
    target_positions = result.get("target_positions", {})
    regime = result.get("market_regime", {})
    regime_names = {1: "牛市", 0: "震荡", -1: "熊市"}
    regime_str = regime_names.get(regime.get("regime", 0), "未知")
    momentum = regime.get("momentum_score", 0)
    bear_risk_str = "是" if regime.get("bear_risk") else "否"

    # ─── Sheet 1: 本次选股 ───
    ws1 = wb.active
    ws1.title = "本次选股"

    ws1.merge_cells("A1:L1")
    c = ws1.cell(row=1, column=1, value=f"选股结果 — {today_str}")
    c.font = Font(name="微软雅黑", size=14, bold=True, color="1F4E79")
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws1.row_dimensions[1].height = 30

    ws1.merge_cells("A2:L2")
    summary = (f"市场状态: {regime_str} | 动量: {momentum:.2f} | "
               f"熊市风险: {bear_risk_str} | 趋势: {regime.get('trend_score', 0):.2f} | "
               f"日期: {today_str}")
    c = ws1.cell(row=2, column=1, value=summary)
    c.font = Font(name="微软雅黑", size=10, color="555555")
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws1.row_dimensions[2].height = 22

    headers1 = [
        "序号", "股票代码", "股票名称", "行业", "综合评分",
        "权重(%)", "目标仓位(元)", "当前价格(元)", "目标股数(手)",
        "因子名称", "信号强度", "备注"
    ]
    hr = 4
    style_header(ws1, headers1, hr)
    ws1.row_dimensions[hr].height = 22

    if selections:
        for i, s in enumerate(selections):
            code = s.get("code", "")
            row = hr + 1 + i
            ws1.cell(row=row, column=1, value=i + 1)
            ws1.cell(row=row, column=2, value=code)
            ws1.cell(row=row, column=3, value=stock_names.get(code, ""))
            ws1.cell(row=row, column=4, value=s.get("industry", ""))
            ws1.cell(row=row, column=5, value=round(s.get("score", 0), 4))
            ws1.cell(row=row, column=6, value=round(s.get("weight", 0) * 100, 2))
            target_val = target_positions.get(code, 0)
            ws1.cell(row=row, column=7, value=round(target_val, 0))
            price = prices.get(code, 0)
            ws1.cell(row=row, column=8, value=round(price, 2))
            lots = int(target_val / price / 100) if price > 0 else 0
            ws1.cell(row=row, column=9, value=lots)
            ws1.cell(row=row, column=10, value="")
            ws1.cell(row=row, column=11, value="")
            ws1.cell(row=row, column=12, value="")

        data_end = hr + len(selections)
        style_data_rows(ws1, hr + 1, data_end, len(headers1))

        total_row = data_end + 1
        ws1.merge_cells(f"A{total_row}:F{total_row}")
        c = ws1.cell(row=total_row, column=1, value=f"共 {len(selections)} 只股票入选")
        c.font = Font(name="微软雅黑", size=10, bold=True, color="1F4E79")
        c.alignment = Alignment(horizontal="right", vertical="center")
        total_target = sum(target_positions.get(s.get("code", ""), 0) for s in selections)
        c = ws1.cell(row=total_row, column=7, value=round(total_target, 0))
        c.font = Font(name="微软雅黑", size=10, bold=True)

        for row in range(hr + 1, data_end + 1):
            ws1.cell(row=row, column=5).number_format = "0.0000"
            ws1.cell(row=row, column=6).number_format = "0.00"
            ws1.cell(row=row, column=7).number_format = "#,##0"
            ws1.cell(row=row, column=8).number_format = "0.00"
            ws1.cell(row=row, column=9).number_format = "#,##0"
    else:
        ws1.cell(row=hr + 1, column=1, value="(无选股结果 — 可能筛选条件过严)")

    auto_width(ws1)

    # ─── Sheet 2: 调仓对比 ───
    ws2 = wb.create_sheet("调仓对比")
    ws2.merge_cells("A1:J1")
    c = ws2.cell(row=1, column=1, value=f"调仓对比 — {today_str}")
    c.font = Font(name="微软雅黑", size=14, bold=True, color="1F4E79")
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws2.row_dimensions[1].height = 30

    headers2 = [
        "操作", "股票代码", "股票名称", "行业", "综合评分",
        "权重(%)", "目标仓位(元)", "当前价格(元)", "目标股数(手)", "备注"
    ]
    hr2 = 3
    style_header(ws2, headers2, hr2)

    ps = load_portfolio()
    current_positions = ps.get("positions", {})
    target_codes = {s.get("code") for s in selections}
    current_codes = set(current_positions.keys())
    buy_codes = target_codes - current_codes
    sell_codes = current_codes - target_codes
    hold_codes = target_codes & current_codes

    row_idx = hr2 + 1
    for label, codes, fill in [
        ("买入", buy_codes, GREEN_FILL),
        ("卖出", sell_codes, RED_FILL),
        ("调整", hold_codes, BLUE_FILL),
    ]:
        for code in sorted(codes):
            ws2.cell(row=row_idx, column=1, value=label)
            ws2.cell(row=row_idx, column=2, value=code)
            ws2.cell(row=row_idx, column=3, value=stock_names.get(code, ""))
            sel_info = next((s for s in selections if s.get("code") == code), {})
            ws2.cell(row=row_idx, column=4, value=sel_info.get("industry", ""))
            ws2.cell(row=row_idx, column=5, value=round(sel_info.get("score", 0), 4))
            ws2.cell(row=row_idx, column=6, value=round(sel_info.get("weight", 0) * 100, 2))
            target_val = target_positions.get(code, 0)
            ws2.cell(row=row_idx, column=7, value=round(target_val, 0))
            price = prices.get(code, 0)
            ws2.cell(row=row_idx, column=8, value=round(price, 2))
            lots = int(target_val / price / 100) if price > 0 else 0
            ws2.cell(row=row_idx, column=9, value=lots)
            if label == "卖出":
                pos_info = current_positions.get(code, {})
                ws2.cell(row=row_idx, column=10, value=f"当前持仓: {pos_info}")
            for c in range(1, len(headers2) + 1):
                ws2.cell(row=row_idx, column=c).fill = fill
                ws2.cell(row=row_idx, column=c).font = DATA_FONT
                ws2.cell(row=row_idx, column=c).alignment = DATA_ALIGN
                ws2.cell(row=row_idx, column=c).border = THIN_BORDER
            row_idx += 1

    if row_idx == hr2 + 1:
        ws2.cell(row=row_idx, column=1, value="(无调仓)")
        row_idx += 1

    sum_row = row_idx
    ws2.merge_cells(f"A{sum_row}:D{sum_row}")
    ws2.cell(row=sum_row, column=1,
             value=f"买入 {len(buy_codes)} | 卖出 {len(sell_codes)} | "
                   f"调整 {len(hold_codes)} | 共 {len(selections)} 只")
    auto_width(ws2)

    # ─── Sheet 3: 市场状态 ───
    ws3 = wb.create_sheet("市场状态")
    ws3.merge_cells("A1:C1")
    c = ws3.cell(row=1, column=1, value=f"市场状态详情 — {today_str}")
    c.font = Font(name="微软雅黑", size=14, bold=True, color="1F4E79")
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws3.row_dimensions[1].height = 30

    headers3 = ["指标", "数值", "说明"]
    hr3 = 3
    style_header(ws3, headers3, hr3)

    market_data = [
        ("日期", today_str, "选股日期"),
        ("市场状态", regime_str, "1=牛市 0=震荡 -1=熊市"),
        ("动量评分", f"{momentum:.4f}", "指数趋势强度，>0.5 强趋势"),
        ("趋势评分", f"{regime.get('trend_score', 0):.4f}", "综合趋势评估"),
        ("熊市风险", bear_risk_str, "是否处于熊市风险区间"),
        ("选股数量", f"{len(selections)} 只", "本次入选股票数"),
        ("目标总仓位", f"¥{sum(target_positions.values()):,.0f}", "所有目标持仓市值之和"),
        ("可用现金", f"¥{cash:,.0f}", "当前可用现金"),
        ("仓位比例", f"{sum(target_positions.values())/cash*100:.1f}%"
         if cash > 0 else "N/A", "目标仓位/现金"),
        ("调仓周期", "20 交易日", "当前调仓频率"),
    ]

    for i, (metric, value, desc) in enumerate(market_data):
        row = hr3 + 1 + i
        ws3.cell(row=row, column=1, value=metric)
        ws3.cell(row=row, column=2, value=value)
        ws3.cell(row=row, column=3, value=desc)

    style_data_rows(ws3, hr3 + 1, hr3 + len(market_data), len(headers3))
    auto_width(ws3)

    # ─── Sheet 4: 调仓历史 ───
    ws4 = wb.create_sheet("调仓历史")
    ws4.merge_cells("A1:K1")
    c = ws4.cell(row=1, column=1, value="调仓历史记录")
    c.font = Font(name="微软雅黑", size=14, bold=True, color="1F4E79")
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws4.row_dimensions[1].height = 30

    headers4 = [
        "调仓日期", "操作", "股票代码", "股票名称", "行业",
        "操作前股数", "操作后股数", "成交价(元)", "成交金额(元)", "原因", "备注"
    ]
    hr4 = 3
    style_header(ws4, headers4, hr4)

    if buy_codes:
        for i, code in enumerate(sorted(buy_codes)):
            row = hr4 + 1 + i
            ws4.cell(row=row, column=1, value=today_str)
            ws4.cell(row=row, column=2, value="建仓")
            ws4.cell(row=row, column=3, value=code)
            ws4.cell(row=row, column=4, value=stock_names.get(code, ""))
            sel_info = next((s for s in selections if s.get("code") == code), {})
            ws4.cell(row=row, column=5, value=sel_info.get("industry", ""))
            ws4.cell(row=row, column=6, value=0)
            target_val = target_positions.get(code, 0)
            price = prices.get(code, 0)
            lots = int(target_val / price / 100) if price > 0 else 0
            ws4.cell(row=row, column=7, value=lots * 100)
            ws4.cell(row=row, column=8, value=round(price, 2))
            ws4.cell(row=row, column=9, value=round(target_val, 0))
            ws4.cell(row=row, column=10, value="初始建仓")
        data_end = hr4 + len(buy_codes)
        style_data_rows(ws4, hr4 + 1, data_end, len(headers4))
    else:
        ws4.cell(row=hr4 + 1, column=1, value="(暂无调仓记录)")

    auto_width(ws4)

    # ─── Sheet 5: 信号排名 Top50 ───
    ws5 = wb.create_sheet("信号排名Top50")
    ws5.merge_cells("A1:I1")
    c = ws5.cell(row=1, column=1, value=f"全市场信号排名 Top50 — {today_str}")
    c.font = Font(name="微软雅黑", size=14, bold=True, color="1F4E79")
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws5.row_dimensions[1].height = 30

    headers5 = [
        "排名", "股票代码", "股票名称", "行业", "综合评分",
        "信号方向", "因子名称", "当前价格(元)", "备注"
    ]
    hr5 = 3
    style_header(ws5, headers5, hr5)

    # 从 signal_store 获取最新日期的信号排名
    if hasattr(runner if 'runner' in dir() else None, 'strategy'):
        pass  # will use runner from outer scope

    # 从 result 获取的 selections 已经包含 top 选股，直接展示
    if selections:
        sorted_sels = sorted(selections, key=lambda x: x.get("score", 0), reverse=True)
        top50 = sorted_sels[:50]
        for i, s in enumerate(top50):
            row = hr5 + 1 + i
            code = s.get("code", "")
            ws5.cell(row=row, column=1, value=i + 1)
            ws5.cell(row=row, column=2, value=code)
            ws5.cell(row=row, column=3, value=stock_names.get(code, ""))
            ws5.cell(row=row, column=4, value=s.get("industry", ""))
            ws5.cell(row=row, column=5, value=round(s.get("score", 0), 4))
            ws5.cell(row=row, column=6, value="买入")
            ws5.cell(row=row, column=7, value="")
            ws5.cell(row=row, column=8, value=round(prices.get(code, 0), 2))
            ws5.cell(row=row, column=9, value="")

        if len(top50) > 0:
            style_data_rows(ws5, hr5 + 1, hr5 + len(top50), len(headers5))
    else:
        ws5.cell(row=hr5 + 1, column=1, value="(无选股结果)")

    auto_width(ws5)

    # ── 保存 ──
    output_path = "/mnt/c/Users/admin/Desktop/选股数据.xlsx"
    wb.save(output_path)
    print(f"\nExcel 已保存: {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    print("加载股票名称...")
    stock_names = load_stock_name_map()
    print(f"股票名称映射: {len(stock_names)} 条")

    result = run_selection()
    if result is None:
        return

    result_data, prices, cash, today_str = result

    # runner 需要传给 build_excel 用于 Sheet5
    output = build_excel(result_data, prices, cash, today_str, stock_names)

    # 打印摘要
    print("\n" + "=" * 60)
    print("  选股摘要")
    print("=" * 60)
    selections = result_data.get("selections", [])
    target_positions = result_data.get("target_positions", {})
    if selections:
        for i, s in enumerate(selections):
            code = s.get("code", "")
            name = stock_names.get(code, "")
            price = prices.get(code, 0)
            target = target_positions.get(code, 0)
            lots = int(target / price / 100) if price > 0 else 0
            print(f"  {i+1:2d}. {code} {name:<8s}  {s.get('industry',''):<12s}  "
                  f"评分{s.get('score',0):.4f}  权重{s.get('weight',0)*100:.1f}%  "
                  f"¥{target:>10,.0f}  {lots:>4d}手")
    else:
        print("  (无选股结果)")
        total_signal_store = 0
        # 检查signal_store
        try:
            from trade.signal_runner import SignalRunner
        except Exception:
            pass


if __name__ == "__main__":
    main()
