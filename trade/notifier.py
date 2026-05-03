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
        regime_val = regime.get("regime", 0)
        regime_emoji = {1: "🐂 牛市", 0: "📊 震荡", -1: "🐻 熊市"}.get(regime_val, "❓ 未知")
        bear = " ⚠️熊市风险" if regime.get("bear_risk") else ""
        momentum = regime.get("momentum_score", 0)
        trend = regime.get("trend_score", 0)

        rebal = rec.get("rebalance_info", {})

        lines = [
            f"## 📈 交易提醒",
            f"",
            f"**{rec.get('date', '')}**  |  上次调仓: {rebal.get('last_rebalance', '-')}  →  下次: {rebal.get('next_rebalance', '-')}",
            f"",
            f"**{regime_emoji}{bear}**  |  动量: `{momentum:+.2f}`  |  趋势: `{trend:+.2f}`",
            f"",
        ]

        # === 买入 ===
        buys = rec.get("buys", [])
        lines.append("### 🟢 买入")
        if buys:
            total_buy = 0
            for b in buys:
                cost = b['estimated_cost']
                total_buy += cost
                ind = b.get('industry', '-')
                score = b.get('score', 0)
                lines.append(f"- **{b['code']}**  ×{b['shares']}股  "
                           f"≈¥{cost:,.0f}  "
                           f"行业: `{ind}`  IC: `{score:.2f}`")
            lines.append(f"")
            lines.append(f"> 买入合计: **¥{total_buy:,.0f}**")
        else:
            lines.append("无")
        lines.append("")

        # === 卖出 ===
        sells = rec.get("sells", [])
        lines.append("### 🔴 卖出")
        if sells:
            total_sell = 0
            for s in sells:
                rev = s['estimated_revenue']
                total_sell += rev
                reason = s.get('reason', '')
                reason_str = f"  _{reason}_" if reason else ""
                lines.append(f"- **{s['code']}**  ×{s['shares']}股  "
                           f"≈¥{rev:,.0f}{reason_str}")
            lines.append(f"")
            lines.append(f"> 卖出合计: **¥{total_sell:,.0f}**")
        else:
            lines.append("无")
        lines.append("")

        # === 继续持有 ===
        holds = rec.get("holds", [])
        if holds:
            lines.append("### 🔵 继续持有")
            for h in holds[:10]:  # 最多显示10只
                lines.append(f"- {h['code']}  ×{h['shares']}股  "
                           f"市值¥{h['current_value']:,.0f}  "
                           f"权重: `{h['weight']:.1%}`")
            if len(holds) > 10:
                lines.append(f"> ...还有 {len(holds) - 10} 只")
            lines.append("")

        # === 汇总 ===
        lines.append("---")
        lines.append(f"💰 现金: ¥{rec.get('cash_before', 0):,.0f}  →  ¥{rec.get('cash_after', 0):,.0f}")
        net = rec.get('total_buy', 0) - rec.get('total_sell', 0)
        lines.append(f"💸 净流出: ¥{net:,.0f}")
        if not rec.get("cash_sufficient", True):
            lines.append(f"")
            lines.append(f"⚠️ **资金不足，请调整买入**")

        return self.send("【交易提醒】需调仓", "\n".join(lines))

    def send_industry_sentiment(self, date_str: str, scores: dict) -> bool:
        """发送每日行业情绪摘要到微信（显示全部行业）

        Args:
            date_str: YYYY-MM-DD 格式的日期
            scores: {industry: sentiment_score}，score 在 [-1.0, 1.0]
        """
        if not scores:
            return False

        sorted_inds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        positive = [(i, s) for i, s in sorted_inds if s > 0.1]
        neutral = [(i, s) for i, s in sorted_inds if -0.1 <= s <= 0.1]
        negative = [(i, s) for i, s in sorted_inds if s < -0.1]

        lines = [
            f"## 每日行业情绪 {date_str}",
            f"覆盖 {len(scores)} 个行业",
            f"",
        ]

        if positive:
            lines.append(f"🟢 看多 ({len(positive)})")
            for ind, score in positive:
                bar = "█" * min(10, max(1, int(score * 10))) + "░" * max(0, 10 - max(1, int(score * 10)))
                lines.append(f"- **{ind}** `{score:+.2f}` {bar}")

        if neutral:
            lines.append(f"")
            lines.append(f"⚪ 中性 ({len(neutral)})")
            for ind, score in neutral:
                lines.append(f"- **{ind}** `{score:+.2f}`")

        if negative:
            lines.append(f"")
            lines.append(f"🔴 看空 ({len(negative)})")
            for ind, score in negative:
                abs_score = abs(score)
                bar = "█" * min(10, max(1, int(abs_score * 10))) + "░" * max(0, 10 - max(1, int(abs_score * 10)))
                lines.append(f"- **{ind}** `{score:+.2f}` {bar}")

        lines.append("")
        lines.append("---")
        lines.append("来源: LLM(DeepSeek) 新闻情绪分析")

        return self.send("每日行业情绪", "\n".join(lines))
