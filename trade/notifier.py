"""微信推送 — Server酱 (sct.ftqq.com)"""
import requests


class Notifier:
    def __init__(self, sckey: str):
        self.sckey = sckey
        self.api = f"https://sctapi.ftqq.com/{sckey}.send"

    def send(self, title: str, content: str) -> bool:
        try:
            resp = requests.post(self.api, data={"title": title, "desp": content}, timeout=10)
            if resp.status_code == 200 and resp.json().get("code") == 0:
                print("微信推送成功")
                return True
            print(f"微信推送失败: {resp.json().get('message', '未知错误')}")
        except Exception as e:
            print(f"微信推送异常: {e}")
        return False

    def send_recommendations(self, rec: dict) -> bool:
        regime = rec.get("market_regime", {})
        regime_names = {1: "牛市", 0: "震荡", -1: "熊市"}
        regime_name = regime_names.get(regime.get("regime", 0), "未知")
        bear_risk = " ⚠熊市风险" if regime.get("bear_risk") else ""

        has_action = rec.get("buys") or rec.get("sells")
        title = "【交易提醒】需调仓" if has_action else "【交易提醒】今日无需调仓"

        lines = [f"市场: {regime_name}{bear_risk}", f"动量: {regime.get('momentum_score', 0):.2f}", ""]

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

        lines += ["", f"买入合计: ¥{rec.get('total_buy', 0):,.0f}",
                   f"卖出合计: ¥{rec.get('total_sell', 0):,.0f}"]
        if not rec.get("cash_sufficient", True):
            lines.append("⚠ **资金不足**")

        return self.send(title, "\n".join(lines))
