"""信号层臂的 yaml 文本补丁 (2026-10-25): 只改 ml.feature_whitelist。

为什么用文本改而不是 yaml round-trip: factor_config.yaml 有大量人工注释,
round-trip 会全部重排 ⇒ diff 爆炸、且注释即文档。文本改保持其余字节不动。

用法: patch_ml_whitelist_1025.py <yaml路径> [--drop-fundamental] [--blend 0.6]
"""
import re
import sys
from pathlib import Path

# 纯基本面特征 (缺失率 13-26%, 是均值填充失真的主要来源)。
# tech_fund_combo 是技术×基本面混合, 保留 (它不是纯基本面)。
FUNDAMENTAL = {
    'fund_pe', 'fund_roe', 'fund_cf_to_profit', 'earnings_quality',
    'profit_growth_accel', 'revenue_growth_accel', 'days_since_report',
    'earnings_yield', 'fund_cf_quality', 'debt_ratio', 'fund_gross_margin',
    'log_total_assets', 'log_revenue', 'log_operating_cf', 'fund_roic',
    'fund_net_margin', 'fund_score', 'fund_profit_growth', 'fund_revenue_growth',
}


def main():
    path = Path(sys.argv[1])
    drop_fund = '--drop-fundamental' in sys.argv
    blend = None
    if '--blend' in sys.argv:
        blend = sys.argv[sys.argv.index('--blend') + 1]
    text = path.read_text(encoding='utf-8')
    original = text

    if drop_fund:
        m = re.search(r'(?m)^(  feature_whitelist:\n)((?:  - .*\n)+)', text)
        if not m:
            raise SystemExit('feature_whitelist block not found')
        block = m.group(2)
        kept = [ln for ln in block.splitlines(keepends=True)
                if ln.strip()[2:].strip() not in FUNDAMENTAL]
        removed = len(block.splitlines()) - len(kept)
        text = text[:m.start(2)] + ''.join(kept) + text[m.end(2):]
        print(f'[patch] feature_whitelist 移除 {removed} 个纯基本面特征, 剩 {len(kept)} 个')

    if blend is not None:
        text, n = re.subn(r'(?m)^(  blend_weight: ).*$', rf'\g<1>{blend}', text)
        print(f'[patch] blend_weight -> {blend} (替换 {n} 处)')

    if text == original:
        raise SystemExit('no change applied')
    path.write_text(text, encoding='utf-8')
    print(f'[patch] 已写入 {path}')


if __name__ == '__main__':
    sys.exit(main())
