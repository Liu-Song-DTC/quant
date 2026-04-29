"""交易确认 — 交互式逐笔确认，支持修改实际成交价/数量"""
import json

from .config import TradeConfig
from .portfolio_state import PortfolioState
from .runner import _get_latest_prices


def confirm_trades(args):
    ps = PortfolioState.load(str(TradeConfig().state_file))
    pending = _collect_pending(args)
    if not pending:
        return

    # 轻量获取参考价
    cfg = TradeConfig()
    codes = [t["code"] for t in pending]
    ref_prices = _get_latest_prices(str(cfg.bt_data_dir), codes)

    confirmed = _interactive_confirm(pending, ref_prices)

    if not confirmed:
        print("\n没有确认的交易")
        return

    _finalize(confirmed, ps)


def _collect_pending(args) -> list:
    """从不同来源收集待确认交易"""
    cfg = TradeConfig()

    if args.interactive or args.from_rec:
        if not cfg.rec_file.exists():
            print("没有上次建议记录，请先运行: python main.py run")
            return []
        with open(cfg.rec_file, "r", encoding="utf-8") as f:
            rec = json.load(f)
        pending = rec.get("trades", [])
        if not pending:
            print("上次建议中没有交易")
        return pending

    if args.buy or args.sell:
        pending = []
        for item in (args.buy or []):
            parts = item.split(":")
            pending.append({"action": "buy", "code": parts[0],
                            "shares": int(parts[1]),
                            "price": float(parts[2]) if len(parts) > 2 else 0})
        for item in (args.sell or []):
            parts = item.split(":")
            pending.append({"action": "sell", "code": parts[0],
                            "shares": int(parts[1]),
                            "price": float(parts[2]) if len(parts) > 2 else 0})
        return pending

    print("用法:")
    print("  python main.py confirm -i              # 交互式逐笔确认（推荐）")
    print("  python main.py confirm --from-rec       # 快速确认上次建议")
    print("  python main.py confirm --buy 600519:100:75.5")
    print("  python main.py confirm --sell 601318:300:48.2")
    return []


def _interactive_confirm(pending: list, ref_prices: dict) -> list:
    """逐笔交互确认"""
    confirmed = []
    names = {"buy": "买入", "sell": "卖出"}

    print(f"\n{'═' * 60}")
    print(f"  交易确认  共 {len(pending)} 笔")
    print(f"  y=确认 | s=跳过 | 输入价格=改价 | 输入数量=改量")
    print(f"{'═' * 60}")

    for i, trade in enumerate(pending):
        action = names.get(trade["action"], trade["action"])
        code = trade["code"]
        ref = ref_prices.get(code, trade.get("price", 0))
        rec_shares = trade["shares"]
        rec_price = trade.get("price", ref)

        print(f"\n  [{i+1}/{len(pending)}] {action} {code}")
        print(f"    建议: ×{rec_shares}股 @ ¥{rec_price:.2f}  参考价: ¥{ref:.2f}")

        while True:
            resp = input(f"    > ").strip().lower()

            if resp in ("y", ""):
                shares, price = rec_shares, rec_price
                if price == 0:
                    try:
                        price = float(input(f"    请输入 {code} 成交价: "))
                    except (ValueError, EOFError):
                        print("    跳过")
                        break
                confirmed.append({"action": trade["action"], "code": code, "shares": shares, "price": price})
                print(f"    ✓ {action} {code} ×{shares} @ ¥{price:.2f} = ¥{shares * price:,.0f}")
                break

            elif resp == "s":
                print(f"    ✗ 跳过 {code}")
                break

            elif resp.isdigit() or (len(resp) > 1 and resp.lstrip("-").isdigit()):
                shares = (abs(int(resp)) // 100) * 100
                if shares <= 0:
                    print(f"    ✗ 跳过 {code}")
                    break
                try:
                    p_in = input(f"    成交价 (回车=¥{ref:.2f}): ").strip()
                    price = float(p_in) if p_in else ref
                except (ValueError, EOFError):
                    price = ref
                if price <= 0:
                    print("    价格无效，跳过")
                    break
                confirmed.append({"action": trade["action"], "code": code, "shares": shares, "price": price})
                print(f"    ✓ {action} {code} ×{shares} @ ¥{price:.2f} = ¥{shares * price:,.0f}")
                break

            else:
                try:
                    price = float(resp)
                    if price > 0:
                        confirmed.append({"action": trade["action"], "code": code, "shares": rec_shares, "price": price})
                        print(f"    ✓ {action} {code} ×{rec_shares} @ ¥{price:.2f} = ¥{rec_shares * price:,.0f}")
                        break
                except ValueError:
                    pass
                print("    无效。y=确认 s=跳过 数字=数量或价格")

    return confirmed


def _finalize(confirmed: list, ps: PortfolioState):
    """汇总确认"""
    names = {"buy": "买入", "sell": "卖出"}
    total_buy = sum(t["shares"] * t["price"] for t in confirmed if t["action"] == "buy")
    total_sell = sum(t["shares"] * t["price"] for t in confirmed if t["action"] == "sell")
    net = total_buy - total_sell

    print(f"\n{'─' * 60}")
    for t in confirmed:
        a = names.get(t["action"], t["action"])
        print(f"  {a} {t['code']} ×{t['shares']} @ ¥{t['price']:.2f} = ¥{t['shares'] * t['price']:,.0f}")
    print(f"\n  买入: ¥{total_buy:,.0f}   卖出: ¥{total_sell:,.0f}   净流出: ¥{net:,.0f}")
    print(f"  当前现金: ¥{ps.cash:,.0f}")
    if ps.cash - net < 0:
        print(f"  ⚠ 操作后现金不足: ¥{ps.cash - net:,.0f}")

    if input("\n  确认记录? (y/N): ").strip().lower() == "y":
        ps.update_after_trade(confirmed)
        print("  交易已记录，组合状态已更新")
    else:
        print("  已取消")
