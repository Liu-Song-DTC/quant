#!/usr/bin/env python
"""市场代码常量: 与深市股票代码冲突的**指数伪码** (2026-10-17)

`raw_data/` 里混着 12 个**指数**目录, 它们的 6 位代码与深市股票代码字面相同,
但内容是上证/深证指数序列(实测 `股票名称` 列):

    000003 Ｂ股指数   000004 工业指数   000005 商业指数   000013 企债指数
    000015 红利指数   000018 180金融    000038 上证金融   000047 上证全指
    000805 A股资源    000851 百发100    000916 300通信    000918 300成长

它们**永不入选**(2600+ 天 `portfolio_selections.csv` 中 0 次出现), 但有两个真实危害:

 ① `bt_execution.py:1977` 只排除 `sh000001` 一个 ⇒ 这些伪码被当作股票放进池子;
 ② 新浪接口按 `sz000018` 抓会抓到**同代码的深市股票**的公司行动 ⇒ 造出假除权。
    实测 000851 被塞入 1.40× 事件, 历史价被改写 **4044 点**。

⇒ 复权重建 / 事件表构建必须**显式排除**, 沿用生产文件。判据无法自动化:
   指数在这份数据里**也有** `换手率`(非零占比 1.000), 无字段可区分, 只能列清单。
"""
import os
import pandas as pd

# 与深市股票代码冲突的指数伪码(见模块 docstring)
INDEX_PSEUDO = frozenset({
    '000003', '000004', '000005', '000013', '000015', '000018',
    '000038', '000047', '000805', '000851', '000916', '000918',
})

# 真指数(代码本身非 6 位数字, 天然不会与股票混): 上证综指 / 中证1000 / 创业板指
INDEX_REAL = ('sh000001', 'sh000852', '399006')


def is_stock_code(code):
    """是否是**真股票**代码(排除 12 个指数伪码 + 3 个真指数)。"""
    return code not in INDEX_PSEUDO and code not in INDEX_REAL


def audit(raw_dir):
    """每次构建前打印被排除的伪码及其名称 —— 厂商若新增伪码, 人能看见。"""
    rows = []
    for c in sorted(INDEX_PSEUDO):
        p = os.path.join(raw_dir, c, 'none.csv')
        nm = '?'
        if os.path.exists(p):
            try:
                nm = pd.read_csv(p, encoding='utf-8-sig', usecols=['股票名称'], nrows=1)['股票名称'].iloc[0]
            except Exception:
                pass
        rows.append((c, nm))
    return rows
