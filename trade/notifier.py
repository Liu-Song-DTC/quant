"""微信推送 — Server酱 (sct.ftqq.com)"""
import requests


class WeChatNotifier:
    """通过Server酱推送微信消息"""

    def __init__(self, sckey: str):
        self.sckey = sckey
        self.api = f"https://sctapi.ftqq.com/{sckey}.send"

    def send(self, title: str, content: str) -> bool:
        """发送消息

        Args:
            title: 消息标题
            content: markdown格式内容

        Returns:
            是否发送成功
        """
        try:
            resp = requests.post(self.api, data={
                "title": title,
                "desp": content,
            }, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 0:
                    print("微信推送成功")
                    return True
                else:
                    print(f"微信推送失败: {data.get('message', '未知错误')}")
            else:
                print(f"微信推送失败: HTTP {resp.status_code}")
        except Exception as e:
            print(f"微信推送异常: {e}")
        return False

    def send_recommendations(self, recommendations: dict, report_path: str = "") -> bool:
        """推送交易建议摘要"""
        rec = recommendations
        regime = rec.get("market_regime", {})
        regime_names = {1: "牛市", 0: "震荡", -1: "熊市"}
        regime_name = regime_names.get(regime.get("regime", 0), "未知")
        bear_risk = "⚠熊市风险" if regime.get("bear_risk") else ""

        title = f"【交易提醒】需调仓" if (rec.get("buys") or rec.get("sells")) else "【交易提醒】今日无需调仓"

        lines = []
        lines.append(f"市场: {regime_name} {bear_risk}")
        lines.append(f"动量: {regime.get('momentum_score', 0):.2f}")
        lines.append("")

        buys = rec.get("buys", [])
        if buys:
            lines.append("**买入:**")
            for b in buys:
                lines.append(f"  {b['code']} ×{b['shares']} ¥{b['estimated_cost']:,.0f}")
        else:
            lines.append("买入: 无")

        sells = rec.get("sells", [])
        if sells:
            lines.append("")
            lines.append("**卖出:**")
            for s in sells:
                lines.append(f"  {s['code']} ×{s['shares']} ¥{s['estimated_revenue']:,.0f} ({s.get('reason', '')})")
        else:
            lines.append("卖出: 无")

        lines.append("")
        lines.append(f"买入合计: ¥{rec.get('total_buy', 0):,.0f}")
        lines.append(f"卖出合计: ¥{rec.get('total_sell', 0):,.0f}")

        if not rec.get("cash_sufficient", True):
            lines.append("")
            lines.append("⚠ **资金不足，需调整买入**")

        if report_path:
            lines.append(f"\n详细报告: {report_path}")

        return self.send(title, "\n".join(lines))

    def test(self) -> bool:
        """发送测试消息"""
        return self.send("测试消息", "实盘交易系统推送测试 ✓")
