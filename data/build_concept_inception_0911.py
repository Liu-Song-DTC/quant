#!/usr/bin/env python3
"""2026-09-11 阶段6c: 构建概念成立日表 concept_inception.pkl (PIT gate)

数据源优先级:
1. THS 概念指数首根K线日 (/tmp/ths_inception.pkl, 292概念, 硬数据)
   - 精确同名 144 个 + 后缀别名匹配
2. concept_hist.pkl min-date (59个EM概念指数, 硬数据; 2023-01-03=窗口底, 视为"早于该日"→不设gate)
3. 业绩预告概念: 名称确定性规则 (2025三季报预增 → 2025-10-01 等)
4. 内部知识表: FUTURE清单(阶段6b armA剥的85个)中无硬数据者, 用保守成立日
   (偏早: 早gate=少剥合法使用; 偏晚=残留前视。取公开资料公认月份)
5. 关键词扫描: 全部504概念扫新题材词, 命中但无日期者列入WARN人工核对清单
输出: data/concept_inception.pkl {concept: 'YYYY-MM-DD'} (仅含>2020-12-01的gate;
缺失=全窗口有效)。北交所无关。串行轻量, 不碰信号链。
"""
import os
import re
import pickle
import pandas as pd

ROOT = '/mnt/d/quant'
EM_MAP = pickle.load(open(f'{ROOT}/data/stock_concept_map.pkl', 'rb'))
EM_CONCEPTS = set()
for cs in EM_MAP.values():
    EM_CONCEPTS.update(cs)
print(f'EM概念: {len(EM_CONCEPTS)}')

inc = {}   # 最终 gate 表
src = {}   # 来源追踪

# ---------- 1) THS 硬数据 ----------
ths = pickle.load(open('/tmp/ths_inception.pkl', 'rb'))
ths_dates = {k: pd.Timestamp(v) for k, v in ths.items()}
# 别名: EM常用后缀与THS名互查 (THS名 → EM名)
ALIASES = {
    'AIGC': 'AIGC概念', 'CPO': 'CPO概念', 'ChatGPT': 'ChatGPT概念', 'Sora': 'Sora概念',
    'Kimi': 'Kimi概念', 'DeepSeek': 'DeepSeek概念', 'Chiplet': 'Chiplet概念',
    'HBM': 'HBM概念', 'CoWoS': 'CoWoS', '5.5G': '5.5G概念', '6G': '6G概念',
    'GLP-1': 'GLP-1概念', 'EDR': 'EDR概念', 'NMN': 'NMN概念', 'MR': 'MR混合现实',
    'AI PC': 'AIPC', 'AIGC算力': '算力概念', '华为昇腾': '华为昇腾',
    '英伟达': '英伟达概念', '小米汽车': '小米汽车', '超导': '超导概念',
    'Web3.0': 'Web3.0', 'C2M': 'C2M概念', 'IGBT': 'IGBT概念', 'MLOps': 'MLOps概念',
    'PET铜箔': 'PET铜箔', 'POE胶膜': 'POE胶膜', '光伏概念': '光伏概念',
}
n_ths = 0
for em_c in EM_CONCEPTS:
    cands = [em_c, em_c.replace('概念', ''), em_c.replace('指数', '')]
    hit = None
    for c in cands:
        if c in ths_dates:
            hit = c
            break
    if hit is None:
        hit = ALIASES.get(em_c)
        if hit and hit in ths_dates:
            pass
        elif hit:
            hit = None
    if hit:
        d = ths_dates[hit]
        if d > pd.Timestamp('2020-12-01'):
            inc[em_c] = d.strftime('%Y-%m-%d')
            src[em_c] = f'ths:{hit}'
            n_ths += 1
print(f'THS twins gate: {n_ths}')

# ---------- 2) concept_hist min-date ----------
# 注意: concept_hist数据底=2023-01-03(39/60概念min-date停在该日, 是数据深度下限
# 不是成立日 — 5G/半导体/人工智能等老概念被误gate教训)。严格晚于数据底才可信。
hist = pickle.load(open(f'{ROOT}/data/concept_hist.pkl', 'rb'))
n_hist = 0
for name, df in hist.items():
    if name not in inc:
        d = df['date'].min()
        if d > pd.Timestamp('2023-01-04'):   # >数据底2023-01-03: 真实成立日
            inc[name] = d.strftime('%Y-%m-%d')
            src[name] = f'hist:{d.date()}'
            n_hist += 1
print(f'concept_hist gate: {n_hist}')

# ---------- 3) 业绩预告确定性规则 ----------
QMAP = {'一季报': '-04-01', '中报': '-07-01', '三季报': '-10-01'}
pat = re.compile(r'^(\d{4})(年报|一季报|中报|三季报)(预增|预减|扭亏|首亏)$')
n_rpt = 0
for c in EM_CONCEPTS:
    m = pat.match(c)
    if not m:
        continue
    y, typ = int(m.group(1)), m.group(2)
    if typ == '年报':
        d = pd.Timestamp(f'{y+1}-01-01')
    else:
        d = pd.Timestamp(f'{y}{QMAP[typ]}')
    if d > pd.Timestamp('2020-12-01'):
        inc[c] = d.strftime('%Y-%m-%d')
        src[c] = f'rpt:{m.group(0)}'
        n_rpt += 1
print(f'业绩预告规则 gate: {n_rpt}')

# ---------- 4) 内部知识表 (FUTURE清单无硬数据者, 保守偏早) ----------
KNOW = {
    # AI大模型系
    'AI智能体': '2024-02-01', 'AI眼镜': '2024-11-01', 'AI手机': '2024-03-01',
    'AI制药（医疗）': '2024-02-01', 'AI医疗': '2024-02-01', 'AI语料': '2024-03-01',
    'AI应用': '2024-02-01', 'AI办公': '2024-03-01', 'AI教育': '2024-03-01',
    'AI安全': '2024-03-01', 'AI服务器': '2023-03-01', 'AIPC': '2024-01-01',
    'AI芯片': '2023-03-01', '大模型概念': '2023-03-01', '文心一言': '2023-03-01',
    '通义千问': '2023-04-01', '讯飞星火': '2023-05-01', '智谱AI': '2023-03-01',
    '文生视频': '2024-02-01', '多模态AI': '2023-11-01',
    # 算力/硬件
    '算力概念': '2023-02-01', '算力租赁': '2023-10-01', '英伟达产业链': '2023-03-01',
    '液冷服务器': '2023-06-01', '液冷概念': '2023-06-01', '液冷温控': '2023-10-01',
    '液冷IDC': '2023-04-01', '液冷超充': '2024-03-01', '硅光子': '2020-12-01',
    '先进封装': '2020-12-01', '玻璃基板': '2024-05-01', 'HBM概念': '2023-03-01',
    'CoWoS': '2023-07-01', 'Chiplet概念': '2022-08-01',
    # 应用/终端
    '人形机器人': '2023-05-01', '机器人执行器': '2023-11-01', '灵巧手': '2025-03-01',
    '低空经济': '2024-03-01', '飞行汽车': '2023-10-01', '车路云': '2024-06-01',
    '萝卜快跑': '2024-07-01', 'Robotaxi': '2024-07-01', '商业航天': '2024-01-01',
    '卫星互联网': '2020-12-01', '卫星通信': '2020-12-01', '低轨卫星': '2020-12-01',
    '通感一体化': '2024-03-01', '5.5G概念': '2023-10-01', '6G概念': '2020-12-01',
    '星闪概念': '2023-08-01', '量子科技': '2020-12-01', '量子计算': '2020-12-01',
    '室温超导': '2023-07-01', '超导概念': '2020-12-01', '可控核聚变': '2023-11-01',
    '深海科技': '2025-03-01', '脑机接口': '2023-05-01',
    # 能源/材料
    '固态电池': '2022-11-01', '半固态电池': '2022-01-01', 'BC电池': '2023-09-01',
    '钙钛矿电池': '2022-06-01', '石英砂概念': '2022-06-01', '合成生物': '2024-04-01',
    # 医药
    '减肥药': '2023-08-01', 'GLP-1概念': '2023-09-01', '司美格鲁肽': '2023-09-01',
    # 内容/消费
    '微短剧': '2023-11-01', '短剧游戏': '2023-10-01', '出海概念': '2023-05-01',
    # 政策/产业
    '数据要素': '2022-12-01', '数据确权': '2022-12-01', '数字中国': '2023-02-01',
    '新质生产力': '2024-01-01', '新型工业化': '2023-09-01',
    # 空间计算系
    'MR混合现实': '2023-06-01', '空间计算': '2023-06-01', '苹果MR': '2023-05-01',
    'VisionPro': '2023-06-01',
    # 二轮扫漏 (2021后成立且在map, THS页1/硬数据未覆盖)
    'AIGC概念': '2022-11-01', 'CPO概念': '2023-02-01', 'Kimi概念': '2024-03-01',
    'DeepSeek概念': '2025-01-01', '短剧互动游戏': '2023-11-01', '车联网(车路云)': '2024-06-01',
    '东数西算': '2022-02-01', 'SPD概念': '2023-08-01', '复合集流体': '2022-08-01',
    '户外露营': '2022-04-01', '抗菌面料': '2022-12-01', '麒麟电池': '2022-06-01',
    'TOPCon电池': '2022-08-01', '汽车一体化压铸': '2022-06-01', '电子后视镜': '2023-02-01',
    '熔盐储能': '2022-08-01', '第四代半导体': '2022-11-01', '冰雪经济': '2022-01-01',
    '谷子经济': '2024-11-01', '首发经济': '2024-12-01', '反内卷概念': '2025-07-01',
    '味蕾经济': '2025-03-01', '科创板做市商': '2022-05-01', '科创板做市股': '2022-05-01',
    '荣耀概念': '2023-11-01', '化债(AMC)概念': '2023-10-01', '新型城镇化': '2022-04-01',
    '混合现实': '2023-06-01', '轮毂电机': '2021-08-01', '空间站概念': '2021-04-01',
    '鸿蒙概念': '2021-05-01', '百度概念': '2021-03-01', '辅助生殖': '2021-04-01',
    '婴童概念': '2021-06-01', '职业教育': '2021-07-01',
}
n_know = 0
for c, d in KNOW.items():
    if c in EM_CONCEPTS and c not in inc:
        inc[c] = d
        src[c] = 'know'
        n_know += 1
print(f'内部知识 gate: {n_know}')

# ---------- 5) 关键词扫描 (漏网清单) ----------
KWS = ['AI', '大模型', '人形', '低空', '商业航天', '算力', '数据要素', '数据确权',
       '液冷', '钙钛矿', '固态', '短剧', '微短剧', '减肥', 'GLP', '司美', '脑机',
       '量子', '卫星', '6G', '星闪', 'HBM', '玻璃基板', 'CoWoS', 'Chiplet', '硅光',
       '超导', '核聚变', '飞行汽车', '新质', 'DeepSeek', 'CPO', 'AIGC', 'AIPC',
       'MR', '空间计算', '萝卜快跑', 'Robotaxi', '车路云', '通感', '文生', '多模态',
       'Kimi', '星火', '通义', '文心', '智谱', '5.5G', '小米汽车', 'BC电池', 'ChatGPT',
       'Sora', '英伟达', '深海', '灵巧手', '合成生物', 'AI手机', 'AI眼镜', '智能体']
warn = [c for c in sorted(EM_CONCEPTS - set(inc))
        if any(k in c for k in KWS)]
print(f'\n[WARN] 新题材关键词命中但无gate日期 ({len(warn)}):')
for c in warn:
    print(f'  {c}')

# ---------- 输出 ----------
out = dict(sorted(inc.items()))
with open(f'{ROOT}/data/concept_inception.pkl', 'wb') as f:
    pickle.dump(out, f)
print(f'\n=== 写入 data/concept_inception.pkl: {len(out)} 个gate '
      f'(ths={n_ths} hist={n_hist} rpt={n_rpt} know={n_know}) ===')
# 摘要: gate落在各年的概念数
ys = pd.Series([pd.Timestamp(v).year for v in out.values()])
print(ys.value_counts().sort_index())
