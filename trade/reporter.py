"""报告输出 — 终端彩色 + markdown文件"""
import os
from datetime import date
from pathlib import Path


REGIME_NAMES = {1: "牛市", 0: "震荡", -1: "熊市"}
REGIME_COLORS = {1: "\033[32m", 0: "\033[33m", -1: "\033[31m"}
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


class Reporter:
    def __init__(self, report_dir: str = "reports/"):
        self.report_dir = Path(report_dir)

    def print_report(self, recommendations: dict, portfolio_summary: dict):
        """终端输出彩色报告"""
        rec = recommendations
        regime = rec.get("market_regime", {})
        regime_val = regime.get("regime", 0)
        regime_name = REGIME_NAMES.get(regime_val, "未知")
        bear_risk = regime.get("bear_risk", False)
        momentum = regime.get("momentum_score", 0)

        print(f"\n{'═' * 60}")
        color = REGIME_COLORS.get(regime_val, "")
        print(f"  交易建议  {date.today()}  市场: {color}{regime_name}{RESET}", end="")
        if bear_risk:
            print(f"  {RED}⚠ 熊市风险{RESET}", end="")
        print(f"  动量: {momentum:.2f}")
        print(f"{'═' * 60}")

        # 买入
        buys = rec.get("buys", [])
        if buys:
            print(f"\n{BOLD}{GREEN}【买入】{RESET}")
            for b in buys:
                print(f"  {b['code']:<8} ×{b['shares']:>4}股  "
                      f"¥{b['estimated_cost']:>10,.0f}  "
                      f"@¥{b['price']:.2f}  "
                      f"IC:{b.get('score', 0):.2f}  "
                      f"行业:{b.get('industry', '-')}")
        else:
            print(f"\n{GREEN}【买入】 无{RESET}")

        # 卖出
        sells = rec.get("sells", [])
        if sells:
            print(f"\n{BOLD}{RED}【卖出】{RESET}")
            for s in sells:
                print(f"  {s['code']:<8} ×{s['shares']:>4}股  "
                      f"¥{s['estimated_revenue']:>10,.0f}  "
                      f"@¥{s['price']:.2f}  "
                      f"原因:{s.get('reason', '-')}")
        else:
            print(f"\n{RED}【卖出】 无{RESET}")

        # 持有
        holds = rec.get("holds", [])
        if holds:
            print(f"\n{BOLD}{CYAN}【继续持有】{RESET}")
            for h in holds:
                print(f"  {h['code']:<8} {h['shares']:>4}股  "
                      f"市值¥{h['current_value']:>10,.0f}  "
                      f"权重:{h['weight']:.1%}  "
                      f"行业:{h.get('industry', '-')}")

        # 汇总
        print(f"\n{'─' * 60}")
        total_buy = rec.get("total_buy", 0)
        total_sell = rec.get("total_sell", 0)
        cash_before = rec.get("cash_before", 0)
        cash_after = rec.get("cash_after", 0)
        sufficient = rec.get("cash_sufficient", True)

        print(f"  预计买入: ¥{total_buy:>12,.0f}")
        print(f"  预计卖出: ¥{total_sell:>12,.0f}")
        print(f"  当前现金: ¥{cash_before:>12,.0f}")
        cash_color = GREEN if sufficient else RED
        print(f"  操作后:   {cash_color}¥{cash_after:>12,.0f}{RESET}", end="")
        if not sufficient:
            print(f"  {RED}⚠ 资金不足，需调整买入{RESET}", end="")
        print()

        # 组合摘要
        if portfolio_summary:
            print(f"\n{'─' * 60}")
            print(f"  组合总资产: ¥{portfolio_summary.get('total_value', 0):>12,.0f}")
            print(f"  持仓市值:   ¥{portfolio_summary.get('market_value', 0):>12,.0f}")
            print(f"  现金:       ¥{portfolio_summary.get('cash', 0):>12,.0f}")
            print(f"  持仓数:     {len(portfolio_summary.get('positions', []))}")

        print(f"\n{'═' * 60}")

        if not buys and not sells:
            print(f"\n  {YELLOW}今日无需调仓{RESET}")

    def save_report(self, recommendations: dict, portfolio_summary: dict) -> str:
        """保存markdown报告"""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        report_date = date.today()
        filepath = self.report_dir / f"{report_date}.md"

        rec = recommendations
        regime = rec.get("market_regime", {})
        regime_name = REGIME_NAMES.get(regime.get("regime", 0), "未知")

        lines = [
            f"# 交易建议 {report_date}",
            f"",
            f"**市场状态**: {regime_name} | "
            f"动量: {regime.get('momentum_score', 0):.2f} | "
            f"熊市风险: {'是' if regime.get('bear_risk') else '否'}",
            f"",
        ]

        # 买入
        buys = rec.get("buys", [])
        lines.append("## 买入")
        if buys:
            lines.append("| 代码 | 数量 | 金额 | 价格 | 行业 |")
            lines.append("|------|------|------|------|------|")
            for b in buys:
                lines.append(f"| {b['code']} | {b['shares']} | ¥{b['estimated_cost']:,.0f} | "
                           f"¥{b['price']:.2f} | {b.get('industry', '-')} |")
        else:
            lines.append("无")

        # 卖出
        sells = rec.get("sells", [])
        lines.append("")
        lines.append("## 卖出")
        if sells:
            lines.append("| 代码 | 数量 | 金额 | 价格 | 原因 |")
            lines.append("|------|------|------|------|------|")
            for s in sells:
                lines.append(f"| {s['code']} | {s['shares']} | ¥{s['estimated_revenue']:,.0f} | "
                           f"¥{s['price']:.2f} | {s.get('reason', '-')} |")
        else:
            lines.append("无")

        # 持有
        holds = rec.get("holds", [])
        lines.append("")
        lines.append("## 继续持有")
        if holds:
            lines.append("| 代码 | 数量 | 市值 | 权重 | 行业 |")
            lines.append("|------|------|------|------|------|")
            for h in holds:
                lines.append(f"| {h['code']} | {h['shares']} | ¥{h['current_value']:,.0f} | "
                           f"{h['weight']:.1%} | {h.get('industry', '-')} |")

        # 汇总
        lines.append("")
        lines.append("## 汇总")
        lines.append(f"- 预计买入: ¥{rec.get('total_buy', 0):,.0f}")
        lines.append(f"- 预计卖出: ¥{rec.get('total_sell', 0):,.0f}")
        lines.append(f"- 当前现金: ¥{rec.get('cash_before', 0):,.0f}")
        lines.append(f"- 操作后: ¥{rec.get('cash_after', 0):,.0f}")
        if not rec.get("cash_sufficient", True):
            lines.append(f"- **⚠ 资金不足**")

        # 组合
        if portfolio_summary:
            lines.append("")
            lines.append("## 组合状态")
            lines.append(f"- 总资产: ¥{portfolio_summary.get('total_value', 0):,.0f}")
            lines.append(f"- 持仓市值: ¥{portfolio_summary.get('market_value', 0):,.0f}")
            lines.append(f"- 现金: ¥{portfolio_summary.get('cash', 0):,.0f}")

        content = "\n".join(lines) + "\n"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"报告已保存: {filepath}")
        return str(filepath)
