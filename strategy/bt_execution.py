# ⚠️ 回测入口 — 修改因子选择/信号生成/组合构建逻辑时,
#    务必同步更新 core/live_init.py 的 init_live_engine/update_live_engine
#    确保回测与实盘等价。
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import os
from tqdm import tqdm
import math
from collections import defaultdict
import multiprocessing
from functools import partial
import gc
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
import ctypes

from core.strategy import Strategy
from core.fundamental import FundamentalData
from core.signal_engine import SignalEngine
from core.factor_preparer import prepare_factor_data, _FACTOR_CODE_FILES, _yaml_stripped_digest
from core.signal_store import SignalStore
from core.config_loader import load_config
from core.monitor import monitor, get_logger
from core.pipeline_logger import plog
from core.industry_mapping import INDUSTRY_KEYWORDS, build_fine_industry_map
from core.market_regime_detector import MarketRegimeDetector
from core.stock_pool import get_stock_pool, get_exclusion_set
from core.cmap_hist import resolve_hist_path as _resolve_cmap_hist

# 情绪分析集成
try:
    from sentiment.orchestrator import SentimentOrchestrator
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# 加载配置
config = load_config()

# 回测参数 - 从配置文件读取
CASH = config.get('backtest.cash', 100000.0)
COMMISSION = config.get('backtest.commission', 0.0001)  # 万1佣金
STAMP_TAX = config.get('backtest.stamp_tax', 0.0005)  # 万5扁平默认 — 9/22成本审计后由 _vectorized_backtest 内 _STAMP_ARR 日期分档取代, 仅作yaml兼容读取
PERC = config.get('backtest.slippage', 0.001)
REBALANCE_DAYS = config.get('backtest.rebalance_days', 20)

# A股涨跌停限制数据（预计算，供BacktraderExecution使用）
_LIMIT_DATA = {}   # {(code, date): 'up' | 'down'}

# 安静模式开关: 默认全速(按配置并行); 显式设 STRATEGY_QUIET=1 或非0 = 安静(2进程降噪)
NUM_WORKERS = config.get('backtest.num_workers', 2)
if os.environ.get('STRATEGY_QUIET', '0') != '0':
    NUM_WORKERS = 2

def _malloc_trim(pad=0):
    """将 Python 已释放但未归还 OS 的内存归还给内核（Linux only）。

    在 gc.collect() 之后调用，可显著降低进程 RSS，避免 fork 时子进程继承
    inflated 内存地址空间导致 OOM。
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(ctypes.c_int(pad))
    except Exception as e:
        import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()  # 非 Linux / 权限不足时静默跳过

# 回测日期范围过滤 (加速测试: fromdate='2024-01-01', todate='2026-05-30')
FROMDATE = config.get('backtest.fromdate', None)
TODATE = config.get('backtest.todate', None)

# E-B1实验(2026-09-05, 已回退不采纳): 曾在此加SIGNAL_WARMUP_DAYS=120暖机窗口,
# 让2021Q1有真因子(默认数据从FROMDATE起喂, 前60bar=信号引擎V41兜底→Q1零买入)。
# 实测: Q1如预期恢复买入(1/14起selected=6), 但2021年22.83%→19.73%(3月白马崩盘),
# 且路径依赖级联拖累全程 → 726,364/190.55%/1.1239 vs 基线818,884/227.55%/1.2546,
# 四项全输。结论: Q1空仓是意外保护, 保持原样。若再试需配Q1特有风险控制。

# 实验钩子: 股票池截止日覆盖 (仅实验用, 默认None=跟随TODATE)
# 用途: 隔离"池子变化 vs 数据追加"对回测的影响, 例 POOL_TODATE=2026-08-28
_POOL_TODATE_OVERRIDE = os.environ.get('POOL_TODATE') or None

# E-B1v2实验钩子(2026-09-07, 已否决不采纳): 信号生成暖机数据窗口(天), 默认0=关。
# 给worker喂FROMDATE-N天数据, 使2021Q1指标暖机(默认前60bar=V41兜底→Q1零买入)。
# E-B1v1(120天, 无风险控制)已否决: Q1恢复买入但2021 -3.1pp+级联拖累全程。
# v2(120天+portfolio层b1v2_q1_half半仓)实测:
#   9/4态 906,528/262.61%/1.3160/15.89% vs 基线743,186 四指标全胜(+163,342)
#   9/2态 1,148,060/359.22%/1.5877/16.70% vs 基线1,131,929 三胜一负
#          (MDD 14.60→16.70恶化+2.10pp, 2022回撤15.9%为恶化来源)
#   铁律(双态四指标均提升)不通过 → 否决留档默认关。若再试: Q1权重bracket
#   (0.6~0.75)或2022路径风控。此env影响信号内容→sidecar指纹协议须含此值
#   (见_signals_stale与L1945附近sidecar写入处, 协议已保留)。
_B1V2_WARMUP_DAYS = int(os.environ.get('B1V2_WARMUP_DAYS', '0') or 0)


def _pool_todate():
    """股票池截止日: 实验覆盖优先, 否则跟随回测TODATE"""
    return _POOL_TODATE_OVERRIDE if _POOL_TODATE_OVERRIDE else TODATE


def _load_pool_membership():
    """0f日历池成员映射 (pool_calendar=quarterly|monthly|daily时): {boundary_ts: set(codes)}.

    off模式返回None (现状as-of todate池行为)。边界覆盖FROMDATE-730d(早盘因子
    训练日期)到TODATE — quarterly=全部季度末, monthly=全部月末(0f-v3 arm A),
    daily=全部日历日(实盘同构: 每日用当日可得数据重算池, 零入池滞后)。
    缓存键=K线数据指纹+stock_pool.py代码态+模式+边界列表+参数 —
    任何变化→重算(~分钟级)。同数据态下缓存命中→秒级。
    """
    _mode = config.get('stock_pool.pool_calendar', 'off')
    if _mode not in ('quarterly', 'monthly', 'daily'):
        return None
    import hashlib
    from core.stock_pool import (_quarter_boundaries, _monthly_boundaries,
                                 _daily_boundaries, get_pool_membership_map)
    from core.factor_preparer import _data_fingerprint
    earliest = pd.Timestamp(FROMDATE) - pd.Timedelta(days=730)
    if _mode == 'daily':
        boundaries = _daily_boundaries(earliest, pd.Timestamp(TODATE))
    elif _mode == 'monthly':
        boundaries = _monthly_boundaries(earliest, pd.Timestamp(TODATE))
    else:
        boundaries = _quarter_boundaries(earliest, pd.Timestamp(TODATE))
    _all_map = {}
    for _item in os.listdir(DATA_PATH):
        if _item.endswith('_qfq.csv'):
            _all_map[_item[:-8]] = DATA_PATH + _item
        elif _item.endswith('_hfq.csv'):
            _all_map[_item[:-8]] = DATA_PATH + _item
    _fp = _data_fingerprint(_all_map, None, [])
    _sp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core', 'stock_pool.py')
    _st = os.stat(_sp)
    # 0f-v3松弛参数 (arm B/C): 缓存键含relax参数, 任一变化→重算membership
    _rf = config.get('stock_pool.pool_relax_floor', 1.0)
    _rm = config.get('stock_pool.pool_relax_momentum', None)
    _key = hashlib.md5(
        f"{_fp}|{_st.st_mtime_ns}|{_st.st_size}|{_mode}|"
        f"{[str(_b.date()) for _b in boundaries]}|"
        f"{config.get('stock_pool.bse_exclude', True)}|{_rf}|{_rm}".encode('utf-8')).hexdigest()[:12]
    print(f"日历池membership({_mode}): 边界{len(boundaries)}个 "
          f"({boundaries[0].date()}..{boundaries[-1].date()}), "
          f"relax={_rf}/{_rm}, 缓存键={_key}")
    return get_pool_membership_map(boundaries, cache_key=_key,
                                   relax_floor=_rf, relax_momentum=_rm)

# 数据路径 - 从配置文件读取，默认相对于策略目录（而非 CWD）
_STRATEGY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_STRATEGY_DIR)
# 2026-10-16: QUANT_DATA_PATH 研究钩子(复权口径臂) — 默认空 = 逐位零改动。
# 换数据源必须重生成信号, 故 DATA_PATH 已计入 _signal_code_fingerprint。
# **归一化是必要的**: 指纹是**字符串**级 — `.../backtrader_data` 与
# `.../backtrader_data/` 指向同一目录却指纹不同 ⇒ 显式传生产路径(少个斜杠)
# 会白跑一次 78min 信号重生成, 且指纹与锚点不可比 (实测 7d13263b vs 8b9e99b5)。
# 归一化后**强制补回尾斜杠**: L125/127/816 是 `DATA_PATH + item` 字符串拼接,
# 去掉尾斜杠会拼出 `...backtrader_data000001_qfq.csv` ⇒ 目录扫描静默空集。
DATA_PATH = os.path.normpath(os.environ.get('QUANT_DATA_PATH') or config.get(
    'paths.data', os.path.join(_PROJECT_DIR, 'data/stock_data/backtrader_data/'))
).rstrip(os.sep) + os.sep
FUNDAMENTAL_PATH = config.get('paths.fundamental', os.path.join(_PROJECT_DIR, 'data/stock_data/fundamental_data/'))


def _signal_code_fingerprint() -> str:
    """信号代码指纹(细化版, 2026-09-06): 决定信号CSV内容的代码/配置指纹。

    锚点危机后第一版(factor_preparer._SIGNAL_CODE_FILES + 整个factor_config.yaml)
    的细化, 解决两个问题:
    ① yaml只哈希非portfolio节 — portfolio节=组合层(portfolio.py消费,
       signal_engine/ml_predictor/factor_preparer均不读), 计入指纹使portfolio
       参数实验(如E-H5)每次改yaml都触发78min信号重生成, 纯浪费;
    ② 纳入config/quarterly_factors/*.yaml — E-D2教训: 季度标定权重强绑定
       信号内容(信号=按季度权重给因子加权), 旧门禁对它不可见, 改权重必须
       手动删CSV。现在自动失效。
    本函数只在信号复用门禁使用, 不影响因子缓存键(缓存键用factor_preparer
    自己的_code代码指纹 — 两者覆盖不同, 因子缓存不关心信号引擎)。
    """
    import hashlib
    import re
    h = hashlib.md5()
    base = os.path.dirname(os.path.abspath(__file__))
    # 代码文件: 因子9文件(yaml单独处理) + 信号引擎 + ML + 本入口
    for rel in [f for f in _FACTOR_CODE_FILES if f != 'config/factor_config.yaml'] + [
            'core/signal_engine.py',
            'core/ml_predictor.py',
            'bt_execution.py',
    ]:
        p = os.path.join(base, rel)
        try:
            st = os.stat(p)
            h.update(f"{rel}|{st.st_mtime_ns}|{st.st_size};".encode('utf-8'))
        except OSError:
            h.update(f"{rel}|MISSING;".encode('utf-8'))
    # factor_config.yaml: 剥掉组合/执行层节后哈希 (共用factor_preparer的
    # _yaml_stripped_digest — 豁免节清单单一来源, 避免两处漂移)。
    _yaml_digest = _yaml_stripped_digest(os.path.join(base, 'config', 'factor_config.yaml'))
    h.update(f"config/factor_config.yaml|{_yaml_digest};".encode('utf-8'))
    # 季度标定权重: 信号权重强绑定 (E-D2教训), 全部23个季度文件计入
    _qdir = os.path.join(base, 'config', 'quarterly_factors')
    try:
        for _qn in sorted(os.listdir(_qdir)):
            _qp = os.path.join(_qdir, _qn)
            _st = os.stat(_qp)
            h.update(f"config/quarterly_factors/{_qn}|{_st.st_mtime_ns}|{_st.st_size};".encode('utf-8'))
    except OSError:
        h.update('config/quarterly_factors|MISSING;'.encode('utf-8'))
    # 2026-10-16: 概念映射计入指纹(**内容级**, 非 mtime)。
    # 教训(假臂, 实测): `_signals_stale` 的数据新鲜度检查是**单向**的 — 只问
    # "有没有比信号 CSV 更新的数据文件"。把概念映射换成一份**更旧**的
    # (冻结快照 / 版本回滚 / QUANT_CONCEPT_MAP 指向历史版本) 对门禁完全不可见
    # ⇒ 静默复用旧信号 ⇒ 臂的 NAV/md5 与基线**逐字节相同**却看不出来
    # (pit_map0531 只跑 9min vs pit_ident 84min, 两臂数字一模一样 = 假臂)。
    # 用**内容 md5** 而非 mtime: 概念映射是信号输入(concept_heat 的成分股归属),
    # 内容相同 ⇒ 信号必然相同(复用正确, 顺带省一次 78min 重生成),
    # 内容不同 ⇒ 必须重生成。两个方向都对。
    _cmap_p = os.environ.get('QUANT_CONCEPT_MAP') or os.path.join(
        _PROJECT_DIR, 'data', 'stock_concept_map.pkl')
    try:
        with open(_cmap_p, 'rb') as _cf:
            _cmap_digest = hashlib.md5(_cf.read()).hexdigest()
        h.update(f"concept_map|{_cmap_p}|{_cmap_digest};".encode('utf-8'))
    except OSError:
        h.update(f"concept_map|{_cmap_p}|MISSING;".encode('utf-8'))

    # 2026-10-16: 版本化概念映射(QUANT_CONCEPT_MAP_HIST)也计入指纹 — 同上, 内容级。
    # 版本化模式下 concept_heat 的成分股名单来自该文件而**不是** stock_concept_map.pkl,
    # 不记账 ⇒ 换 history 文件时指纹不变 ⇒ 静默复用上一臂的信号(同类假臂, 已踩过)。
    _cmap_hist_p = _resolve_cmap_hist()
    if _cmap_hist_p:
        try:
            with open(_cmap_hist_p, 'rb') as _hf:
                _hh = hashlib.md5(_hf.read()).hexdigest()
            h.update(f"cmap_hist|{_cmap_hist_p}|{_hh};".encode('utf-8'))
        except OSError:
            h.update(f"cmap_hist|{_cmap_hist_p}|MISSING;".encode('utf-8'))

    # 2026-10-16: **数据源路径**也计入指纹 (复权口径臂引入 QUANT_DATA_PATH 钩子)。
    # 同类假臂防线: 换 DATA_PATH 而指纹不变 ⇒ 因子仍从**旧数据源**算出 ⇒
    # 信号被静默复用 ⇒ 臂只换了成交价不换选股 = 半假臂。路径变了必须重生成。
    h.update(f"data_path|{DATA_PATH};".encode('utf-8'))
    return h.hexdigest()[:8]


def _latest_data_date():
    """K线最新交易日(指数文件末行, 'YYYY-MM-DD') — 信号新鲜度的内容级基准."""
    _idx = os.path.join(DATA_PATH, 'sh000001_qfq.csv')
    try:
        _df = pd.read_csv(_idx, usecols=['datetime'])
        return str(_df['datetime'].iloc[-1])[:10] if len(_df) else ''
    except Exception:
        return ''


def _signals_stale(signals_csv):
    """信号CSV是否比任何数据文件旧 (数据刷新后必须重新生成信号).
    2026-09-04: 旧行为无条件复用既有CSV, 数据全量刷新后曾混用旧信号
    (被杀旧代码实验残留CSV + 9/4新数据 → 混搭基线)。同数据态下portfolio类
    实验仍走复用快路径 (数据未变则信号CSV不旧).
    2026-09-06: 增加信号代码指纹门禁 — 797,071锚点危机(执行层代码漂移
    指纹不可见)教训: 生成信号的代码变了但CSV复用 → 混搭代码态不可复现.
    """
    import time as _t
    sig_mtime = os.path.getmtime(signals_csv)
    # 信号代码指纹门禁: 生成CSV时代码指纹存于sidecar, 当前代码指纹不一致
    # (含bt_execution/signal_engine/ml_predictor, 见factor_preparer._SIGNAL_CODE_FILES)
    # → 强制重生成。sidecar缺失(旧时代CSV/首次运行)同样强制。
    _sidecar = os.path.join(os.path.dirname(os.path.abspath(signals_csv)),
                            '.signal_code_fp')
    try:
        with open(_sidecar, 'r') as _f:
            _fp_saved = _f.read().strip()
    except OSError:
        _fp_saved = ''
    # E-B1v2: 暖机env影响信号内容, 代码指纹(mtime+size)不可见 → sidecar记录
    # "fp|warmup_days", 任一不一致→重生成。旧格式sidecar(无'|')视为warmup未知→重生成
    _fp_saved_warmup = ''
    if '|' in _fp_saved:
        _fp_saved, _fp_saved_warmup = _fp_saved.split('|', 1)
    _fp_now = _signal_code_fingerprint()
    _warmup_now = str(_B1V2_WARMUP_DAYS)
    if _fp_saved != _fp_now or _fp_saved_warmup != _warmup_now:
        print(f"信号CSV代码态过期 (生成时={_fp_saved or '无记录'}/warmup={_fp_saved_warmup or '?'} "
              f"当前={_fp_now}/warmup={_warmup_now}), 重新生成信号")
        return True
    # 2026-10-14: todate推进必须触发重生成 — 内容级检查(旧门禁只看mtime+代码指纹).
    # 教训: 锚链实验(todate=10-12)晚于每日流时, 每日流把todate推进到10-13却因数据
    # 文件未变而复用止于10-12的CSV → 10/13实盘用过期一天信号出单(用户实盘发现
    # "价格是昨天的")。规则: CSV最后日期必须覆盖 min(todate, K线最新交易日) —
    # 用min避免周末/节假日(数据未更新)时无谓重生成。
    _expected = _latest_data_date()
    _todate = str(config.get('backtest.todate', '') or '')[:10]
    if _todate and (not _expected or _todate < _expected):
        _expected = _todate
    if _expected:
        try:
            _sig_max = str(pd.read_csv(signals_csv, usecols=['date'])['date'].max())[:10]
        except Exception as _e:
            print(f"信号CSV日期读取失败({_e}), 重新生成信号")
            return True
        if _sig_max < _expected:
            print(f"信号CSV止于 {_sig_max} < 应覆盖 {_expected} (todate/K线最新日), 重新生成信号")
            return True
    # 2026-09-12 收窄(#54): 信号层仅消费4个另类文件 — 经消费路径核验, signal_engine
    # 只调get_northbound_signal(northbound_daily)/get_margin_signal(margin_daily)/
    # get_dragon_tiger_signal(dragon_tiger_history+dragon_tiger); 其余(yjyg/unlock/
    # reduction_records/reduction_plans/holdernum/增持/回购/研报/大宗/margin_detail)
    # 均为组合层输入, 组合层每次运行实时读取, 不改变信号内容 → 不触发信号重生成。
    # 旧整目录根: 任一另类文件mtime更新(如yjyg每日下载)即78min信号重生成churn。
    _SIGNAL_ALT_FILES = ['northbound_daily.pkl', 'margin_daily.pkl',
                         'dragon_tiger_history.pkl', 'dragon_tiger.pkl']
    roots = [DATA_PATH, FUNDAMENTAL_PATH,
             *[os.path.join(_PROJECT_DIR, 'data/alternative_data', fn)
               for fn in _SIGNAL_ALT_FILES],
             os.path.join(_PROJECT_DIR, 'data/concept_hist.pkl'),
             os.environ.get('QUANT_CONCEPT_MAP') or os.path.join(
                 _PROJECT_DIR, 'data/stock_concept_map.pkl'),
             os.path.join(_PROJECT_DIR, 'data/concept_inception.pkl')]
    # 2026-10-17: 版本化概念映射**默认开启**, 故恒计入新鲜度。
    # (它确实是信号输入了; 且 stock_concept_map.pkl 本就在 roots 里且每天变,
    #  所以"每天重生成一次信号"这件事的成本没有变化。)
    _ch = _resolve_cmap_hist()
    if _ch:
        roots.append(_ch)
    newest, newest_what = 0, ''
    for root in roots:
        if os.path.isdir(root):
            for dirpath, _dirs, files in os.walk(root):
                for fn in files:
                    p = os.path.join(dirpath, fn)
                    try:
                        mt = os.path.getmtime(p)
                    except OSError:
                        continue
                    if mt > newest:
                        newest, newest_what = mt, p
        elif os.path.exists(root):
            mt = os.path.getmtime(root)
            if mt > newest:
                newest, newest_what = mt, root
    if sig_mtime < newest:
        print(f"信号CSV过期 (构建 {_t.ctime(sig_mtime)} 早于 数据最新 {_t.ctime(newest)}: "
              f"{os.path.basename(newest_what)}), 重新生成信号")
        return True
    return False


# ===== ML chunk跨日缓存 (A+C项 2026-10-10) =====
# 机制: chunk网格是_pred_dates上的index网格(range(0,N,retrain_freq)),
# 新交易日只把新日期追加到_pred_dates末尾 → 既有chunk的chunk_dates/
# first_pred_date/purge_end/train_mask全部不变 → 训练产物(模型+预测)可跨日复用。
# 失效键=数据指纹(K线+基本面+题材+两融, mtime_ns+size)+信号代码指纹(含
# ml_predictor/bt_execution/yaml剥壳摘要/季度标定) — 任一变化→全miss→全量重训,
# 失效方向安全(只多算一次, 不会错用旧值)。与_signals_stale门禁同源。
# 落盘顺序: preds parquet(原子) → model(原子) → manifest最后引用 → 中途被杀
# 只产生无人引用的孤儿chunk文件, 下次运行miss重训, 不产生静默错用。
_ML_CHUNK_CACHE_DIR = os.path.join(_STRATEGY_DIR, 'rolling_validation_results', 'ml_chunk_cache')


def _ml_chunk_cache_fp(stock_file_map, fundamental_data, boundaries) -> dict:
    """ML缓存失效指纹 — 按chunk边界日期切分的K线内容指纹, 返回{date_str: hex}.

    关键设计(2026-10-10): 不能全用mtime — 每日数据刷新会重写全部K线文件
    (mtime+size全变)但历史行不变, mtime基指纹会让跨日复用永远失效; 固定
    "去末行"hash也不行 — append后窗口平移(旧末行进入hash体)指纹仍变。
    正确切分=按chunk依赖边界: chunk k的数据=日期≤last_pred_date的所有K线行,
    新交易日追加的bar日期>既有边界→该chunk指纹稳定→跨日命中; 历史修正
    (qfq重算/概念回修, 10/9重锚教训: 数据源确实修历史)改的是≤边界的行→失效重训。
    实现: 每文件每行md5一次(~7.5M行/全库), 按日期≤边界聚合为每边界一个摘要。
    成本~10-15s/run, 循环前一次。
    """
    import hashlib
    _bkeys = sorted({str(pd.Timestamp(b).date()) for b in boundaries})
    _mds = {_k: hashlib.md5() for _k in _bkeys}
    for _p in sorted(set(stock_file_map.values())):
        try:
            _raw = open(_p, 'rb').read()
        except OSError:
            for _k in _bkeys:
                _mds[_k].update(f"{_p}|MISSING;".encode('utf-8'))
            continue
        _lines = _raw.split(b'\n')
        if _lines and not _lines[-1]:
            _lines.pop()
        if not _lines:
            continue
        _dates = np.array([_l.split(b',', 1)[0] for _l in _lines])
        _lh = np.array([hashlib.md5(_l).digest() for _l in _lines], dtype='S16')
        for _k in _bkeys:
            # 日期YYYY-MM-DD零填充 → 字典序=时间序
            _sel = _dates <= _k.encode('utf-8')
            _mds[_k].update(_lh[_sel].tobytes())
    return {_k: _m.hexdigest()[:12] for _k, _m in _mds.items()}


def _ml_chunk_cache_aux_fp(fundamental_data, config=None) -> str:
    """ML缓存辅助指纹: 非日期可切分输入(基本面/题材pkl)+ML相关config节+信号代码指纹。
    任一变化→全部chunk失效(这些输入非append-only, 变了就是真变)。
    config节覆盖feature_whitelist/min_val_ic/模型超参等键中未含项 —
    否则改config会静默服务陈旧缓存模型(跨日正确性hazard)。
    """
    import hashlib, json
    _h = hashlib.md5()
    if config is not None:
        try:
            _cc = getattr(config, 'config', config)
            for _sec in ('ml', 'margin_features', 'dynamic_factor'):
                _sec_v = _cc.get(_sec) if isinstance(_cc, dict) else None
                if _sec_v is not None:
                    _h.update(f"{_sec}=".encode('utf-8'))
                    _h.update(json.dumps(_sec_v, sort_keys=True, default=str).encode('utf-8'))
                    _h.update(b";")
        except Exception:
            _h.update(b"configfp-ERR;")
    else:
        _h.update(b"noconfig;")
    if fundamental_data is not None and getattr(fundamental_data, 'data_path', None):
        for _root, _dirs, _files in os.walk(fundamental_data.data_path):
            for _f in sorted(_files):
                _fp = os.path.join(_root, _f)
                try:
                    _st = os.stat(_fp)
                    _h.update(f"{_fp}|{_st.st_mtime_ns}|{_st.st_size};".encode('utf-8'))
                except OSError:
                    _h.update(f"{_fp}|MISSING;".encode('utf-8'))
    for _f in ('concept_hist.pkl', 'stock_concept_map.pkl', 'concept_inception.pkl'):
        # 研究钩子: QUANT_CONCEPT_MAP 覆盖(默认空=生产路径) — 换路径必须换指纹, 否则信号静默复用
        _p = (os.environ.get('QUANT_CONCEPT_MAP')
              if _f == 'stock_concept_map.pkl' else None) or os.path.join(_PROJECT_DIR, 'data', _f)
        try:
            _st = os.stat(_p)
            _h.update(f"{_p}|{_st.st_mtime_ns}|{_st.st_size};".encode('utf-8'))
        except OSError:
            _h.update(f"{_p}|MISSING;".encode('utf-8'))
    return f"{_h.hexdigest()[:12]}|{_signal_code_fingerprint()}"


def _ml_chunk_cache_margin_fp(boundaries) -> dict:
    """margin(两融)按chunk边界切分的内容指纹: {date_str: hex}。

    _ml_df经merge_asof(backward, allow_exact_matches=False)接入两融特征 —
    历史行的margin取值只由≤该行日期的margin决定, 新追加margin日不改变
    既有行 → 按边界切分hash与K线同构; 历史修正旧margin行→失效。加载
    失败→{'*': 'ERR'}使全部chunk miss(安全方向, 与_ml_df降级一致)。
    """
    import hashlib
    _out = {}
    _margin_p = os.path.join(_PROJECT_DIR, 'data', 'alternative_data', 'margin_daily.pkl')
    try:
        import pickle
        with open(_margin_p, 'rb') as _f:
            _mdf = pickle.load(_f)
        if not hasattr(_mdf, 'columns') or 'date' not in _mdf.columns:
            return {str(pd.Timestamp(_b).date()): 'ERR' for _b in boundaries}
        for _b in boundaries:
            _k = str(pd.Timestamp(_b).date())
            _sub = _mdf[_mdf['date'] < pd.Timestamp(_k)]
            _h = hashlib.md5()
            _h.update(pd.util.hash_pandas_object(_sub, index=False).values.tobytes())
            _out[_k] = _h.hexdigest()[:8]
    except Exception:
        return {str(pd.Timestamp(_b).date()): 'ERR' for _b in boundaries}
    return _out


def _ml_chunk_manifest_path() -> str:
    return os.path.join(_ML_CHUNK_CACHE_DIR, 'manifest.json')


def _ml_chunk_load_manifest() -> dict:
    import json
    try:
        with open(_ml_chunk_manifest_path(), 'r') as _f:
            return json.load(_f)
    except Exception:
        return {}


def _ml_chunk_save_manifest(_m: dict) -> None:
    import json
    try:
        os.makedirs(_ML_CHUNK_CACHE_DIR, exist_ok=True)
        _tmp = _ml_chunk_manifest_path() + '.tmp'
        with open(_tmp, 'w') as _f:
            json.dump(_m, _f)
        os.replace(_tmp, _ml_chunk_manifest_path())
    except Exception as _e:
        print(f"[ML缓存] manifest写入失败(忽略, 下次全量重训): {_e}")


def _ml_chunk_load(_key: str, _config):
    """命中→(model_or_None, outcome, val_ic, preds_df); 未命中/任何异常→None(回退全量训练).

    model_or_None=None表示该chunk复用上一chunk的有效模型(原逻辑不切换
    _last_good_predictor); outcome∈{trained_good, reuse_prev, reuse_forced,
    reuse_chain_exceeded} — 与训练路径的val_ic裁决分支一一对应, 重放保持
    _val_ics/_reused_chunks/_consecutive_reuse演化逐位一致。
    """
    _m = _ml_chunk_load_manifest()
    _e = _m.get(_key)
    if not _e:
        return None
    try:
        from core.ml_predictor import MLFactorPredictor
        _model = None
        if _e.get('model'):
            _model = MLFactorPredictor(_config.config)
            _model.load_model(os.path.join(_ML_CHUNK_CACHE_DIR, _e['model']))
            # load_model对缺失文件静默降级(跳过不存在的ensemble成员), 必须验证
            # 完整加载 — 否则半损模型静默进入信号生成 (正确性门禁, 2026-10-10)
            if not getattr(_model, '_ensemble_models', None) or \
                    not getattr(_model, 'feature_cols', None):
                raise RuntimeError(f'缓存模型不完整(文件缺失?): {_e["model"]}')
        _preds = pd.read_parquet(os.path.join(_ML_CHUNK_CACHE_DIR, _e['preds']))
        return (_model, _e['outcome'], _e.get('val_ic'), _preds,
                os.path.join(_ML_CHUNK_CACHE_DIR, _e['model']) if _e.get('model') else None)
    except Exception as _ex:
        print(f"[ML缓存] chunk加载失败(回退训练): {_ex}")
        return None


def _ml_chunk_save(_key: str, _idx: int, _outcome: str, _val_ic, _model,
                   _codes_arr, _dates_arr, _preds_arr) -> None:
    """chunk产物落盘: 先preds(原子)→model(原子)→manifest最后引用.

    reuse_prev/reuse_chain_exceeded不存模型(model=None): 重放时沿用上一chunk
    有效模型, 与原逻辑不切换_last_good_predictor一致。
    """
    _m = _ml_chunk_load_manifest()
    _ent = {'idx': _idx, 'outcome': _outcome, 'val_ic': _val_ic, 'model': None}
    try:
        os.makedirs(_ML_CHUNK_CACHE_DIR, exist_ok=True)
        _pfile = f'chunk_{_idx}_preds.parquet'
        _pdf = pd.DataFrame({'code': _codes_arr, 'date': _dates_arr, 'ml_pred': _preds_arr})
        _pdf.to_parquet(os.path.join(_ML_CHUNK_CACHE_DIR, _pfile), index=False)
        _ent['preds'] = _pfile
        if _model is not None:
            _mfile = f'chunk_{_idx}_model.json'
            _model.save_model(os.path.join(_ML_CHUNK_CACHE_DIR, _mfile))
            _ent['model'] = _mfile
        _m[_key] = _ent
        _ml_chunk_save_manifest(_m)
    except Exception as _e:
        print(f"[ML缓存] chunk保存失败(忽略, 下次全量重训): {_e}")


# 全局变量用于 worker 进程
_worker_engine = None
_worker_use_dynamic = False

# 模块级默认值（函数内通过 global 声明赋值，后处理代码读取）
dynamic_factor_stats = {'hit': 0, 'miss': 0, 'factor_names': {}}
use_dynamic = False
precomputed_cache = None
_ml_total_preds = 0
_ml_val_ic = None


def _build_concept_industry_codes(stock_codes, min_stocks=20):
    """从概念板块映射构建 industry_codes，与离线标定对齐。

    替代 prepare_factor_data 的关键词匹配（20大类），使用 stock_concept_map.pkl
    的精确概念板块，与 calibrate_industry_regime / factor_config.yaml 的分组方式完全一致。

    Returns:
        dict: {concept_name: [code1, code2, ...]}
    """
    import pickle
    # 研究钩子: QUANT_CONCEPT_MAP 覆盖概念映射路径(默认空=生产路径, 逐位零改动)
    concept_map_path = os.environ.get('QUANT_CONCEPT_MAP') or os.path.join(
        _PROJECT_DIR, 'data', 'stock_concept_map.pkl')
    if not os.path.exists(concept_map_path):
        print("[WARNING] stock_concept_map.pkl 不存在，回退到关键词映射")
        return {}

    with open(concept_map_path, 'rb') as f:
        raw = pickle.load(f)

    STYLE_KW = ['融资融券', '深股通', '沪股通', '富时罗素', '标准普尔', 'MSCI',
                '创业板综', '机构重仓', 'QFII', '破增发', '破发股', '昨日高',
                '中证500', '深成500', '中盘股', '小盘股', '央国企改革',
                '西部大开发', '年报预增', '专精特新', '上证380', 'HS300',
                '微盘股', '百元股', '大盘股', '小盘成长', '小盘价值',
                '转债标的', '长江三角', '深圳特区', '破净股', '创投',
                # v2: 过滤投资风格/主题标签，只保留产业类概念
                '养老金', '社保重仓', '基金重仓', '券商重仓', '保险重仓',
                '茅指数', '央视50', '大盘价值', '大盘成长', '中盘价值', '中盘成长',
                '价值股', '成长股', '红利股', '低市盈率', '高市盈率',
                '证金', '汇金', '国资云', '央企改革', '地方国资',
                '上证180', '上证50', '沪深300', '中证100', '中证1000',
                '深证100', '创业板指', '科创50', '科创100',
                'AB股', 'AH股', 'B股', 'ST股',
                '昨日涨停', '昨日首板', '昨日炸板', '昨日连板',
                '次新股', '预增', '预减', '扭亏', 'IPO受益', '壳资源',
                '参股券商', '参股银行', '参股保险', '参股期货', '参股新三板',
                '券商概念', '微盘精选', '百元股', '低安全分', '破发', '回购', '举牌',
                '长期破净', '高股息', '股权转让', '送转', '高送转',
                '新高', '权重股', '行业龙头', '近期新高', '历史新高', '百日新高',
                # v3: 回测日志中高频出现的非产业概念/风格标签
                '中特估', '中字头', '上证180_', '上证50_', '央视50_', '深证100R',
                '低价股', '周期股', '证金持股', '一带一路',
                '创业成份', 'REITs概念', '最近多板', '东方财富热股',
                '内贸流通', '成渝特区', '湖北自贸',
                ]
    stock_codes_set = set(stock_codes)
    concept_to_codes = defaultdict(list)
    for code, concepts in raw.items():
        if code not in stock_codes_set:
            continue
        filtered = [c for c in concepts if not any(kw in c for kw in STYLE_KW)]
        for c in filtered:
            concept_to_codes[c].append(code)

    result = {
        c: clist for c, clist in concept_to_codes.items()
        if len(clist) >= min_stocks
    }
    print(f"概念板块映射: {len(result)} 个概念 (min_stocks={min_stocks}), "
          f"覆盖 {len(set(c for clist in result.values() for c in clist))} 只股票")
    return result


def _init_worker(fundamental_path, stock_codes, use_dynamic, industry_codes, factor_cache, all_dates, regime_df,
                 ml_model_path=None, ml_preds_path=None, rank_parquet_path=None, quarter_configs=None):
    """Worker 进程初始化函数 (spawn 模式)

    每个 worker 独立创建 FundamentalData 和 ML 预测器，避免 fork 导致的
    OpenMP/XGBoost 死锁问题。
    ML 预测量改为按股票代码按需从 parquet 读取（不再加载 69MB pickle/进程）。
    因子排名同样按需从 parquet 读取。
    """
    global _worker_engine, _worker_use_dynamic, _worker_ml_preds_path, _worker_rank_parquet_path
    _worker_use_dynamic = use_dynamic
    _worker_ml_preds_path = ml_preds_path  # 只记路径, 按需读取
    _worker_rank_parquet_path = rank_parquet_path

    # 性能(2026-10-09): 主进程预解析的季度因子配置直接注入, 跳过worker侧YAML解析
    # (cProfile: _load_quarter_config 23次×0.66s≈15s/worker 首股税, 8 workers并发CPU峰)
    if quarter_configs:
        import core.signal_engine as _se_mod
        _se_mod._QUARTER_CONFIG_CACHE.update(quarter_configs)

    _worker_engine = SignalEngine()

    # spawn 模式：每个 worker 独立创建 FundamentalData
    if fundamental_path is not None and stock_codes is not None:
        from core.fundamental import FundamentalData
        _worker_fd = FundamentalData(fundamental_path, stock_codes=stock_codes)
        _worker_engine.set_fundamental_data(_worker_fd)

    if regime_df is not None:
        _worker_engine.set_market_regime(regime_df)

    if use_dynamic and factor_cache is not None and all_dates is not None:
        _worker_engine.set_industry_mapping(industry_codes)
        _worker_engine.dynamic_factor_selector.set_factor_cache(factor_cache, all_dates)

    # 内存(2026-10-09): worker侧_ml_predictor零消费者 (信号侧消费走_ml_predictions/parquet),
    # fixed模式(生产)跳过加载 → 每worker省~144MB (3-ensemble模型), 8 workers共省~1.2GB
    if use_dynamic and ml_model_path is not None and os.path.exists(ml_model_path):
        try:
            from core.ml_predictor import MLFactorPredictor
            worker_ml = MLFactorPredictor()
            worker_ml.load_model(ml_model_path)
            _worker_engine.set_ml_predictor(worker_ml)
        except Exception as e:
            import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()


def _generate_stock_signal_worker(args):
    """Worker 函数：为一个股票生成信号 — 直接从文件读取，避免 pickle 传大数据"""
    global _worker_engine, _worker_use_dynamic, _worker_diag_reported, _worker_ml_preds_path, _worker_rank_parquet_path
    code, filepath = args
    code = str(code).zfill(6)  # 归一化为6位字符串，确保与所有下游系统一致

    try:
        engine = _worker_engine

        if engine is None:
            engine = SignalEngine()

        # 按需加载本股票的 ML 预测（~500条, <10KB）, 替代 69MB 全量 pickle
        if _worker_ml_preds_path is not None and os.path.exists(_worker_ml_preds_path):
            _stock_preds_df = pd.read_parquet(_worker_ml_preds_path,
                                              filters=[('code', '==', code)])
            if len(_stock_preds_df) > 0:
                _stock_preds = {
                    (str(row['code']).zfill(6), pd.Timestamp(row['date'])): float(row['ml_pred'])
                    for _, row in _stock_preds_df.iterrows()
                }
                engine.set_ml_predictions(_stock_preds)
            else:
                engine.set_ml_predictions({})

        # 按需加载本股票的截面排名数据（~650行 × 50因子, ~130KB）, 用于因子组合评分
        if _worker_rank_parquet_path is not None and os.path.exists(_worker_rank_parquet_path):
            _stock_rank_df = pd.read_parquet(_worker_rank_parquet_path,
                                              filters=[('code', '==', code)])
            if len(_stock_rank_df) > 0:
                _stock_rank_df['date'] = pd.to_datetime(_stock_rank_df['date'])
                engine.set_stock_rank_data(code, _stock_rank_df)
            else:
                engine.set_stock_rank_data(code, None)

        store = SignalStore()
        data = pd.read_csv(filepath, parse_dates=['datetime'])
        # 日期范围过滤（与主进程一致的加速优化）
        # E-B1v2: 暖机窗口=多喂FROMDATE前N天数据供指标暖机(信号仍含暖机行,
        # 回测从FROMDATE起消费, 多余行闲置无害)
        if FROMDATE:
            _from_dt = pd.Timestamp(FROMDATE)
            if _B1V2_WARMUP_DAYS > 0:
                _from_dt = _from_dt - pd.Timedelta(days=_B1V2_WARMUP_DAYS)
            data = data[data['datetime'] >= _from_dt]
        if TODATE:
            data = data[data['datetime'] <= TODATE]
        if len(data) < 60:
            del data
            return (code, {}, {})
        engine.generate(code, data, store)

        if hasattr(engine, 'fundamental_data') and engine.fundamental_data is not None:
            engine.fundamental_data.clear_stock_cache(code)

        # 内存(2026-10-09): engine逐股缓存残留清理 — _bom_cache按(code,date)每股~1330条浮点,
        # 累积至2689股≈600MB/worker (8 workers耗尽16GB→池内reclaim压力每股5.7s vs 顺序1.5s)。
        # 读路径均仅当前code自身(generate内无跨code读, 每股仅处理一次) → 每股后清空恒等且O(1)。
        for _ck_name in ('_bom_cache', '_fd_sorted_cache', '_fd_decline_cache', '_spec_ind_cache'):
            _c = getattr(engine, _ck_name, None)
            if _c is not None:
                _c.clear()

        dyn_fail_snapshot = {}
        if hasattr(engine, '_dyn_fail'):
            dyn_fail_snapshot = dict(engine._dyn_fail)
            engine._dyn_fail = {k: 0 for k in engine._dyn_fail}

        # 收集 worker 的诊断数据（因子选择、买入信号等）
        diag_data = {}
        if hasattr(engine, '_diag') and engine._diag is not None:
            m = engine._diag.metrics
            diag_data = {
                'factor_selection': dict(m.get('factor_selection', {})),
                'buy_signals': m.get('buy_signals', 0),
                'buy_by_buy_point': dict(m.get('buy_by_buy_point', {})),
                'buy_by_factor_family': dict(m.get('buy_by_factor_family', {})),
                'gate_quality_samples': list(m.get('gate_quality_samples', [])),
                'hard_rejects': m.get('hard_rejects', 0),
                'alt_nb': m.get('alt_northbound_hits', 0),
                'alt_mg': m.get('alt_margin_hits', 0),
                'alt_dt': m.get('alt_dragon_tiger_hits', 0),
            }
            # A.6-1(2026-09-22): 季度配置诊断计数器是signal_engine模块级dict, spawn
            # worker内累积后随进程消亡 → 主进程_print_quarter_diag恒空。快照+清零回传。
            try:
                from core import signal_engine as _se_mod
                _qd = {k: int(v) for k, v in _se_mod._QUARTER_DIAG.items() if k != 'printed'}
                _se_mod._QUARTER_DIAG.update({k: 0 for k in _se_mod._QUARTER_DIAG if k != 'printed'})
            except Exception:
                _qd = {}
            if diag_data:
                diag_data['quarter_diag'] = _qd
            engine._diag.reset()

        # 收集 worker 的因子选择统计
        stats_snapshot = {}
        if hasattr(engine, '_stats'):
            stats_snapshot = dict(engine._stats)
            # 类型保持重置(2026-10-09): ic_values等list容器保持list (原{k:0}把list翻转成int)
            engine._stats = {k: ([] if isinstance(v, list) else 0) for k, v in engine._stats.items()}

        # 收集 worker 的 BOM 统计
        bom_snapshot = {}
        if hasattr(engine, '_bom_diag'):
            bd = engine._bom_diag
            bom_snapshot = {
                'total': bd.get('total', 0),
                'hit': bd.get('hit', 0),
                'miss': bd.get('miss', 0),
                'moat': bd.get('moat', 0),
                'sum_score': bd.get('sum_score', 0.0),
                'n_unique': len(bd.get('_unique_codes', set())),
                'n_moat_codes': len(bd.get('_moat_codes', set())),
            }
            # Reset worker BOM counters
            bd['total'] = bd['hit'] = bd['miss'] = bd['moat'] = 0
            bd['sum_score'] = 0.0
            bd['_unique_codes'].clear()
            bd['_moat_codes'].clear()

        result = (code, store._store, dyn_fail_snapshot, diag_data, stats_snapshot, bom_snapshot)
        del data
        del store
        return result
    except Exception as e:
        print(f"[Worker Error] {code}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return (code, {}, {}, {}, {})


def add_data_and_signal(cerebro, strategy, fundamental_data=None, membership_map=None):
    """加载数据、生成信号、添加到cerebro。
    同时预计算涨跌停数据和ST股票集合。

    membership_map: 0f日历池成员映射(见_load_pool_membership) — 非None时
    stock_file_map过滤用union池(任何边界曾入池的码保留全史), 信号消费端按日闸。
    """

    global _LIMIT_DATA, _ml_total_preds, _ml_val_ic
    all_items = os.listdir(DATA_PATH)
    stock_codes = []

    # === Phase 0: 扫描文件构建路径映射（不加载数据，避免 ~200MB stock_data_dict） ===
    stock_file_map = {}  # code → filepath
    _LIMIT_DATA = {}     # 重置涨跌停数据
    _LOAD_COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'change_percent', 'amplitude']
    _DTYPE_MAP = {
        'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
        'volume': 'float32', 'change_percent': 'float32', 'amplitude': 'float32',
    }
    for item in all_items:
        if item.startswith('._'):
            continue
        filepath = DATA_PATH + item
        if item.endswith('_qfq.csv'):
            name = item[:-8]
        elif item.endswith('_hfq.csv'):
            name = item[:-8]
        else:
            continue
        stock_file_map[name] = filepath

    # === 实盘保真臂开关: 科创板权限2026-10-13用户确认已有(出单层剔除撤销, 回测=实盘口径),
    #     QUANT_STAR_EXCLUDE保留作纯反事实工具(量化科创板贡献, 普查已用: −256k p=0.150), 默认关. ===
    if os.environ.get('QUANT_STAR_EXCLUDE') == '1':
        _n_star = len(stock_file_map)
        stock_file_map = {k: v for k, v in stock_file_map.items()
                          if not k.startswith(('688', '689'))}
        print(f"[live-fidelity] 剔除科创板(688/689): {_n_star} -> {len(stock_file_map)} 只")

    # === 股票池过滤（先过滤再预计算，避免读不需要的文件） ===
    stock_pool_enabled = config.get('stock_pool.enabled', True)
    if stock_pool_enabled:
        if membership_map is not None:
            # 0f日历池: union池过滤 — 任何边界曾入池的码都保留全史(掩码/闸在消费端)
            _union = set()
            for _codes in membership_map.values():
                _union |= _codes
            pool_codes = _union | {'sh000001', '000001', 'sh000852', '399006'}
            before_count = len(stock_file_map)
            stock_file_map = {k: v for k, v in stock_file_map.items() if k in pool_codes}
            after_count = len(stock_file_map)
            print(f"股票池过滤(日历union): {before_count} -> {after_count} 只 "
                  f"(任何日历边界曾入池)")
        else:
            stock_pool = get_stock_pool(todate=_pool_todate(),
                                        bse_exclude=config.get('stock_pool.bse_exclude', True))
            pool_codes = stock_pool | {'sh000001', '000001', 'sh000852', '399006'}
            before_count = len(stock_file_map)
            stock_file_map = {k: v for k, v in stock_file_map.items() if k in pool_codes}
            after_count = len(stock_file_map)
            print(f"股票池过滤: {before_count} -> {after_count} 只 (全部通过质量筛选)")
    else:
        print(f"股票池过滤: 已关闭，使用全市场 {len(stock_file_map)} 只股票")

    # 预计算涨跌停数据（流式读取，仅加载 change_percent + amplitude 列，读后即弃）
    print("预计算涨跌停数据（流式读取）...")
    for name, filepath in tqdm(list(stock_file_map.items()), desc="limit data"):
        try:
            data = pd.read_csv(filepath, parse_dates=['datetime'],
                              usecols=['datetime', 'change_percent', 'amplitude'])
        except Exception:
            continue
        if FROMDATE:
            data = data[data['datetime'] >= FROMDATE]
        if TODATE:
            data = data[data['datetime'] <= TODATE]
        if name == 'sh000001' or 'change_percent' not in data.columns:
            continue
        chg = data['change_percent'].values
        amp = data['amplitude'].values if 'amplitude' in data.columns else np.ones(len(data)) * 99
        dates = data['datetime'].values
        for i in range(len(data)):
            cp = chg[i]
            if pd.isna(cp):
                continue
            ap = amp[i] if not pd.isna(amp[i]) else 99
            # Fix: 原ap<3.0过严，大幅漏判真实涨停（开盘-2%拉到+10%振幅12%被漏掉）
            # 涨幅9.5%+在A股必然触及涨停板，振幅只用排除极端尾盘逆转(ap>=20%)
            if cp >= 9.5 and ap < 20.0:
                _LIMIT_DATA[(name, pd.Timestamp(dates[i]).date())] = 'up'
            elif cp <= -9.5 and ap < 20.0:
                _LIMIT_DATA[(name, pd.Timestamp(dates[i]).date())] = 'down'
            elif 4.5 <= cp <= 5.5 and ap < 12.0:  # ST涨停±5%, 放宽振幅
                _LIMIT_DATA[(name, pd.Timestamp(dates[i]).date())] = 'up'
            elif -5.5 <= cp <= -4.5 and ap < 12.0:
                _LIMIT_DATA[(name, pd.Timestamp(dates[i]).date())] = 'down'
        del data
    print(f"涨跌停数据: {len(_LIMIT_DATA)} 条 (涨停={sum(1 for v in _LIMIT_DATA.values() if v=='up')}, "
          f"跌停={sum(1 for v in _LIMIT_DATA.values() if v=='down')})")

    # === 科创板(688xxx) 不再剔除 — 2025-2026年妖股主要集中板块 ===
    star_codes = set()
    before_excl_star = len(stock_file_map)
    print(f"科创板保留: {before_excl_star} 只 (不再排除)")

    # ST股票不再静态排除 — 改为向量化回测中按ST状态时间线逐日过滤(get_st_timeline)
    # 先加载sh000001获取交易日历（所有后续步骤依赖）
    sh000001_file = stock_file_map.get('sh000001') or stock_file_map.get('000001')
    if sh000001_file:
        index_df = pd.read_csv(sh000001_file, parse_dates=['datetime'])
        if FROMDATE:
            index_df = index_df[index_df['datetime'] >= FROMDATE]
        if TODATE:
            index_df = index_df[index_df['datetime'] <= TODATE]
        calendar_index = pd.DatetimeIndex(sorted(index_df['datetime']))
    else:
        # fallback: 从任意一只股票获取日期
        calendar_index = None
        for fp in list(stock_file_map.values())[:1]:
            tmp = pd.read_csv(fp, parse_dates=['datetime'])
            calendar_index = pd.DatetimeIndex(sorted(tmp['datetime']))
            break

    # 处理指数数据 - 生成市场状态 (Fix#6: 多指数风格检测)
    # calendar_index 已在上方从 sh000001 构建
    regime_df = None
    # 加载小盘/成长指数（如果存在）
    small_cap_df = None
    growth_df = None
    # Fix#43 (2026-09-04): 小盘风格输入应为中证1000指数(sh000852)。
    # 原实现误用深市股票 000852(石化机械) — 该文件是股票而非指数, 大小盘判断被单只个股驱动。
    # 优先用 sh000852(Windows下载器补下后生效), 缺失时回退旧行为(保持与旧基线可比)。
    if 'sh000852' in stock_file_map:
        small_cap_df = pd.read_csv(stock_file_map['sh000852'], parse_dates=['datetime'])
    elif '000852' in stock_file_map:
        small_cap_df = pd.read_csv(stock_file_map['000852'], parse_dates=['datetime'])
        print('[WARN] 未找到 sh000852(中证1000), 回退用 000852 股票数据当小盘输入 — 请在Windows下载器补下中证1000指数')
    if '399006' in stock_file_map:
        growth_df = pd.read_csv(stock_file_map['399006'], parse_dates=['datetime'])

    if sh000001_file:
        strategy.generate_market_regime(
            index_df,
            small_cap_df=small_cap_df,
            growth_df=growth_df,
        )
        regime_df = strategy.index_data
        has_sc = small_cap_df is not None
        has_gr = growth_df is not None
        print(f"市场状态数据已生成，共 {len(regime_df)} 条记录"
              f" (中证1000={'✓' if has_sc else '✗'}, 创业板指={'✓' if has_gr else '✗'})")
    # 释放小盘/成长指数数据（已不需要）
    del small_cap_df, growth_df

    # 准备动态因子数据
    factor_mode = config.config.get('factor_mode', 'both')
    factor_df = None
    industry_codes = {}
    all_dates = sorted(calendar_index.tolist()) if calendar_index is not None else []
    # 因子训练需要历史数据: 扩展日期范围覆盖 lookback(250T) + IC warm-up(180T)
    # 430个交易日 ≈ 2年日历日，确保 factor_df 从回测起点就有足够的IC训练窗口
    _raw_sh_path = os.path.join(os.path.dirname(DATA_PATH.rstrip('/')), 'raw_data', '000001', 'qfq.csv')
    if os.path.exists(_raw_sh_path):
        _raw_dates = pd.to_datetime(pd.read_csv(_raw_sh_path, encoding='utf-8-sig', usecols=['日期'])['日期'])
        _early_dates = sorted(d for d in _raw_dates if pd.Timestamp(FROMDATE) - pd.Timedelta(days=730) <= d < pd.Timestamp(FROMDATE))
        all_dates = sorted(set(_early_dates) | set(all_dates))
    # 因子数据计算需要 factor_df: 动态模式(IC选择) 或 ML训练 时需要
    # fixed模式但ML启用时也需计算factor_df, 供ML训练使用
    _ml_enabled = config.config.get('ml', {}).get('enabled', False)
    _need_factor_df = factor_mode != 'fixed' or _ml_enabled
    if _need_factor_df:
        print(f"准备因子数据 (factor_mode={factor_mode})...")
        # 获取股票代码列表（排除指数: 上证/中证1000/创业板指, Fix#43）
        stock_codes = [name for name in stock_file_map.keys()
                       if name not in ('sh000001', 'sh000852', '399006')]
        factor_df, industry_codes, all_dates = prepare_factor_data(
            stock_file_map,
            fundamental_data,
            INDUSTRY_KEYWORDS,
            all_dates,
            NUM_WORKERS,
            membership_map=membership_map
        )
        strategy.set_factor_data(factor_df, industry_codes)
        strategy.set_sector_data({}, industry_codes)
        # ConceptHeat 使用概念板块（与行业映射独立，不覆盖因子选择的行业分组）
        concept_codes = _build_concept_industry_codes(stock_codes, min_stocks=30)
        if concept_codes:
            strategy.set_sector_data(concept_codes, industry_codes)
        print(f"因子模式: {factor_mode}, {len(industry_codes)} 个行业, {len(concept_codes)} 个概念")
    else:
        print(f"跳过IC计算: factor_mode={factor_mode} (fixed模式)")
        # 排除指数 (Fix#43: sh000852/399006 同 sh000001)
        stock_codes = [name for name in stock_file_map.keys()
                       if name not in ('sh000001', 'sh000852', '399006')]
        # 使用关键词构建行业映射（prepare_factor_data 依赖INDUSTRY_KEYWORDS）
        industry_codes = build_fine_industry_map(fundamental_data, stock_codes)
        if industry_codes:
            strategy.set_industry_mapping(industry_codes)
            print(f"行业映射: {len(industry_codes)} 个行业")

    # factor_df 已由 prepare_factor_data 在工作进程中计算完成
    # stock_data_dict 不再存在 — 数据由worker从磁盘按需读取，无需清理
    import gc
    gc.collect()
    _malloc_trim()
    print("因子数据准备完成（流式加载，无 stock_data_dict 内存峰值）")

    # ===================================================================
    # Phase A: ML训练 + 因子预计算（依赖 factor_df，需在 factor_df 释放前完成）
    # 先于 Backtrader datafeed 创建，避免 reindexed 数据(~2GB) 与 factor_df(~4GB) 共存
    # ===================================================================

    # === ML预测层训练 (滚动窗口 Walk-Forward) ===
    ml_config = config.config.get('ml', {})
    _ml_model_path = None
    _ml_preds = {}
    _ml_codes_arr = _ml_dates_arr = _ml_preds_arr = None  # 向量化预测累积 (性能2026-10-09)
    _has_xgb = False
    try:
        import xgboost as _xgb
        _has_xgb = True
    except ImportError:
        pass
    if ml_config.get('enabled', False) and factor_df is not None and _has_xgb:
        try:
            from core.ml_predictor import MLFactorPredictor

            # 两融截面特征接入(仅ML训练/预测用, 不进信号引擎的factor_df;
            # T-1滞后: 融资融券收盘后披露, merge_asof不取当日匹配)
            _ml_df = factor_df
            if config.get('margin_features.enabled', False):
                try:
                    from core.alternative_data import get_provider
                    _mfeat = get_provider().get_margin_feature_frame()
                    if _mfeat is not None and len(_mfeat):
                        _mfeat = _mfeat.copy()
                        _fdf = factor_df.sort_values('date')
                        _mfeat['date'] = pd.to_datetime(_mfeat['date']).astype(_fdf['date'].dtype)
                        _fdf['code'] = _fdf['code'].astype(str)
                        _mfeat['code'] = _mfeat['code'].astype(str)
                        _ml_df = pd.merge_asof(_fdf, _mfeat, on='date', by='code',
                                               direction='backward', allow_exact_matches=False)
                        print(f"[ML] 两融特征已接入: rz_chg5覆盖率 "
                              f"{_ml_df['rz_chg5'].notna().mean() * 100:.1f}%")
                        # ml_predictor按feature_whitelist过滤特征, 两融特征需显式加入才进模型
                        _wl = ml_config.get('feature_whitelist')
                        if _wl is not None:
                            for _c in ('rz_chg5', 'rz_chg20', 'rz_buy_ratio', 'rqyl_chg5'):
                                if _c not in _wl:
                                    _wl.append(_c)
                        del _fdf
                except Exception as e:
                    print(f"[ML] 两融特征接入失败(降级为不用): {e}")
                    _ml_df = factor_df

            _train_window = ml_config.get('train_window_days', 750)
            _retrain_freq = ml_config.get('retrain_frequency', 60)
            _pred_start = pd.Timestamp(ml_config.get('pred_start_date', '2021-01-01'))

            _all_dates = sorted(factor_df['date'].unique())
            _pred_dates = [d for d in _all_dates if d >= _pred_start]

            if len(_pred_dates) == 0:
                print("[ML] 无预测日期, 跳过")
            else:
                print(f"ML滚动窗口: train_window={_train_window}日 "
                      f"retrain_every={_retrain_freq}日 "
                      f"pred_dates={len(_pred_dates)}")

                _regime_map = {}
                if regime_df is not None and 'regime' in regime_df.columns:
                    for _idx, _row in regime_df.iterrows():
                        k = _idx.strftime('%Y-%m-%d') if isinstance(_idx, pd.Timestamp) else str(_idx)
                        _regime_map[k] = int(_row['regime'])

                chunk_starts = list(range(0, len(_pred_dates), _retrain_freq))
                _val_ics = []
                _total_preds = 0
                _last_good_predictor = None
                _reused_chunks = 0
                # 性能(2026-10-09): 预测结果向量化累积为数组列表, 替代3.2M次逐条dict插入
                # (~1GB峰值dict + ~2-4min构建) — 落盘时直接concat建表
                _ml_codes_list, _ml_dates_list, _ml_preds_list = [], [], []
                _fd_date_set = set(factor_df['date'].values)
                # 复用链上限: 连续复用N个chunk后旧模型已陈旧, 强制采用新模型保持适应性
                _max_reuse = int(ml_config.get('max_consecutive_reuse', 3))
                _consecutive_reuse = 0
                # A+C(2026-10-10): ML chunk跨日缓存指纹, 循环前算一次 —
                # K线内容按chunk边界日期(各chunk的last_pred_date)切分hash:
                # 新交易日追加的bar日期>既有边界→指纹稳定→跨日命中;
                # 历史修正(≤边界的行变化)→失效重训。辅助指纹(基本面/题材/代码)
                # + margin按边界切分同理。键含chunk身份。
                _ml_cache_boundaries = [_pred_dates[min(_cs + _retrain_freq, len(_pred_dates)) - 1]
                                        for _cs in chunk_starts]
                _ml_cache_fp = _ml_chunk_cache_fp(stock_file_map, fundamental_data,
                                                  _ml_cache_boundaries)
                _ml_cache_aux = _ml_chunk_cache_aux_fp(fundamental_data, config)
                _ml_cache_margin = _ml_chunk_cache_margin_fp(_ml_cache_boundaries)
                # loop不变量提升 (原在循环内逐chunk重算, 值不变)
                _fp_days = int(config.get('dynamic_factor.forward_period', 10))

                for chunk_idx, chunk_start in enumerate(tqdm(chunk_starts, desc="ML滚动训练")):
                    chunk_end = min(chunk_start + _retrain_freq, len(_pred_dates))
                    chunk_dates = _pred_dates[chunk_start:chunk_end]
                    first_pred_date = chunk_dates[0]

                    # Purged walk-forward: 标签=forward_period日前瞻收益,
                    # 训练窗必须止于first_pred_date前forward_period个交易日,
                    # 否则靠近训练期末的标签窗口越过预测日(回测前视, 实盘拿不到)
                    # all_dates在此是密集交易日历(本函数开头由sh000001日历+早段
                    # 补齐日期构建, prepare_factor_data原样返回), 故_fp_days=10
                    # 个条目=10个交易日purge, 与E-D1采纳基线(830,353×2逐位一致,
                    # chunk0训练样本203,902)及9/5全天各run行为一致。
                    # 2026-09-06 797,071锚点危机定案: 根因是19:42 run的执行层代码
                    # 漂移(E2D2残留ml_predictor/signal_engine态, IC不可复现)+该run
                    # 信号生成期间另类数据三件套下载竞态; purge行本身无罪, 勿再动。
                    # A+C(2026-10-10): 跨日缓存命中路径 — 同数据态+代码态下chunk的
                    # 日期窗/训练窗/模型全不变, 直接重放缓存模型+预测。val_ic裁决链
                    # 状态按outcome重放, 与训练路径逐位一致; 新追加日期(跨日时仅末
                    # chunk)用重放后的有效模型预测并回写缓存。
                    import hashlib as _hl
                    _lb_key = str(pd.Timestamp(chunk_dates[-1]).date())
                    _cache_key = _hl.md5(
                        f"{_ml_cache_fp.get(_lb_key, 'NODIGEST')}|{_ml_cache_aux}|"
                        f"{_ml_cache_margin.get(_lb_key, 'NOMARGIN')}|{chunk_start}|"
                        f"{pd.Timestamp(first_pred_date).strftime('%Y-%m-%d')}|"
                        f"{pd.Timestamp(chunk_dates[-1]).strftime('%Y-%m-%d')}|"
                        f"{_train_window}|{_fp_days}".encode('utf-8')).hexdigest()[:16]
                    _cache_hit = _ml_chunk_load(_cache_key, config)
                    if _cache_hit is not None:
                        _c_model, _c_outcome, _c_val_ic, _c_preds, _c_model_path = _cache_hit
                        if _c_outcome == 'trained_good':
                            _val_ics.append(_c_val_ic)
                            _last_good_predictor = _c_model
                            _consecutive_reuse = 0
                            if _c_model_path:
                                _ml_model_path = _c_model_path
                        elif _c_outcome == 'reuse_forced':
                            _last_good_predictor = _c_model
                            _consecutive_reuse = 0
                        elif _c_outcome in ('reuse_prev', 'reuse_chain_exceeded'):
                            _reused_chunks += 1
                            _consecutive_reuse += 1
                        _c_dates_set = set(np.asarray(_c_preds['date'].values, dtype='datetime64[ns]'))
                        _new_dates = [d for d in chunk_dates
                                      if np.datetime64(pd.Timestamp(d)) not in _c_dates_set]
                        if _new_dates:
                            _c_new_rows = []
                            for _nd in _new_dates:
                                _ndf = _ml_df[_ml_df['date'] == _nd]
                                if len(_ndf) == 0:
                                    continue
                                _nrv = _regime_map.get(_nd.strftime('%Y-%m-%d'), 0) if _regime_map else 0
                                _npr = _last_good_predictor.predict(_ndf, regime_info={'regime': _nrv})
                                _ncd = _ndf['code'].values
                                _c_new_rows.append(pd.DataFrame({
                                    'code': np.asarray([str(_c).zfill(6) for _c in _ncd]),
                                    'date': np.asarray([_nd] * len(_ncd), dtype='datetime64[ns]'),
                                    'ml_pred': np.asarray(_npr)}))
                            if _c_new_rows:
                                _c_preds = pd.concat([_c_preds] + _c_new_rows, ignore_index=True)
                                _ml_chunk_save(_cache_key, chunk_idx, _c_outcome, _c_val_ic,
                                               _c_model if _c_outcome in ('trained_good', 'reuse_forced') else None,
                                               _c_preds['code'].values, _c_preds['date'].values,
                                               _c_preds['ml_pred'].values)
                        # 重放本chunk全部预测行 (最终落盘按(code,date)排序, 顺序无关)
                        if len(_c_preds):
                            _ml_codes_list.append(np.asarray([str(_c) for _c in _c_preds['code'].values]))
                            _ml_dates_list.append(np.asarray(_c_preds['date'].values, dtype='datetime64[ns]'))
                            _ml_preds_list.append(np.asarray(_c_preds['ml_pred'].values))
                        _total_preds += sum(1 for d in chunk_dates if d in _fd_date_set)
                        continue

                    _d0_ts = pd.Timestamp(first_pred_date)
                    _d0_pos = int(np.searchsorted(np.asarray(all_dates), _d0_ts))
                    _purge_end = all_dates[max(0, _d0_pos - _fp_days)]
                    train_start = first_pred_date - pd.Timedelta(days=_train_window)
                    train_mask = (_ml_df['date'] >= train_start) & \
                                 (_ml_df['date'] < _purge_end)
                    train_df = _ml_df[train_mask]

                    if len(train_df) < 50000:
                        print(f"  chunk {chunk_idx}: 训练样本不足({len(train_df)}), 跳过")
                        continue

                    ml_predictor = MLFactorPredictor(config.config)
                    _train_regime = 0
                    if _regime_map:
                        _train_regs = [_regime_map.get(d.strftime('%Y-%m-%d'), 0)
                                       for d in train_df['date'].unique()]
                        _train_regime = int(np.median(_train_regs)) if _train_regs else 0
                    val_ic = ml_predictor.train(train_df, regime_info={'regime': _train_regime})
                    _min_val_ic = ml_config.get('min_val_ic', 0.03)
                    if val_ic is None or val_ic < _min_val_ic:
                        if _last_good_predictor is None:
                            if val_ic is not None and val_ic > 0:
                                print(f"  chunk {chunk_idx}: 验证IC={val_ic:.4f} < {_min_val_ic}, 跳过")
                            continue
                        if _consecutive_reuse < _max_reuse:
                            # 验证失败不能整段跳过: 预测断档会让实盘信号
                            # 前向填充到更早日期的陈旧ML分数(验证IC门是防坏模型,
                            # 不是制造数据空洞). 复用上一有效模型预测本chunk日期.
                            print(f"  chunk {chunk_idx}: 验证IC={val_ic}, "
                                  f"<{_min_val_ic}, 复用上一有效模型")
                            ml_predictor = _last_good_predictor
                            _reused_chunks += 1
                            _consecutive_reuse += 1
                            _chunk_outcome = 'reuse_prev'
                        elif val_ic is not None and val_ic > 0:
                            # 复用链超上限且新模型IC>0: 旧模型已陈旧, 采用新模型
                            print(f"  chunk {chunk_idx}: 验证IC={val_ic:.4f} <{_min_val_ic}, "
                                  f"连续复用{_consecutive_reuse}次达上限, 采用新模型")
                            _last_good_predictor = ml_predictor
                            _consecutive_reuse = 0
                            _chunk_outcome = 'reuse_forced'
                        else:
                            # 新模型IC<=0: 比陈旧的好模型更差, 继续复用并告警
                            print(f"  chunk {chunk_idx}: 验证IC={val_ic}, "
                                  f"复用链已达{_consecutive_reuse}次且新模型无正向IC, 继续复用")
                            ml_predictor = _last_good_predictor
                            _reused_chunks += 1
                            _consecutive_reuse += 1
                            _chunk_outcome = 'reuse_chain_exceeded'
                    else:
                        _val_ics.append(val_ic)
                        _last_good_predictor = ml_predictor
                        _consecutive_reuse = 0
                        _chunk_outcome = 'trained_good'
                        strategy_dir = os.path.dirname(os.path.abspath(__file__))
                        model_dir = os.path.join(strategy_dir, 'models')
                        os.makedirs(model_dir, exist_ok=True)
                        _ml_model_path = os.path.join(model_dir, 'xgb_strategy_model.json')
                        ml_predictor.save_model(_ml_model_path)

                    _n0 = len(_ml_codes_list)
                    for date in chunk_dates:
                        date_df = _ml_df[_ml_df['date'] == date]
                        if len(date_df) == 0:
                            continue
                        _regime_val = _regime_map.get(date.strftime('%Y-%m-%d'), 0) if _regime_map else 0
                        preds = ml_predictor.predict(date_df, regime_info={'regime': _regime_val})
                        codes = date_df['code'].values
                        # 向量化累积 (性能2026-10-09): 逐条dict插入 → 数组append
                        _ml_codes_list.append(np.asarray([str(c).zfill(6) for c in codes]))
                        _ml_dates_list.append([date] * len(codes))
                        _ml_preds_list.append(np.asarray(preds))
                    _total_preds += sum(1 for d in chunk_dates
                                        if d in _fd_date_set)
                    # A+C(2026-10-10): 本chunk产物入缓存(skip不缓存), 下次同态复用
                    _ml_chunk_save(_cache_key, chunk_idx, _chunk_outcome, val_ic,
                                   ml_predictor if _chunk_outcome in ('trained_good', 'reuse_forced') else None,
                                   np.concatenate(_ml_codes_list[_n0:]) if len(_ml_codes_list) > _n0 else np.array([], dtype='<U6'),
                                   np.concatenate(_ml_dates_list[_n0:]) if len(_ml_dates_list) > _n0 else np.array([], dtype='datetime64[ns]'),
                                   np.concatenate(_ml_preds_list[_n0:]) if len(_ml_preds_list) > _n0 else np.array([], dtype='float32'))

                # 向量化合并 (性能2026-10-09): 动态模式主引擎仍需(code,date)->pred dict,
                # fixed模式(生产)仅需数组落盘 — 原3.2M条目dict不再在fixed模式构建
                if _ml_codes_list:
                    _ml_codes_arr = np.concatenate(_ml_codes_list)
                    _ml_dates_arr = np.concatenate(_ml_dates_list)
                    _ml_preds_arr = np.concatenate(_ml_preds_list)
                    _ml_total_preds = len(_ml_codes_arr)
                    if factor_mode != 'fixed':
                        # 显式pd.Timestamp键 (np.concatenate会把Timestamp列表转成datetime64,
                        # 与引擎内部的pd.Timestamp查找键hash一致)
                        _ml_preds = dict(zip(zip(_ml_codes_arr,
                                                 [pd.Timestamp(d) for d in _ml_dates_arr]),
                                             _ml_preds_arr))
                else:
                    _ml_total_preds = 0
                _ml_val_ic = float(np.mean(_val_ics)) if _val_ics else 0.0
                _true_skip = len(chunk_starts) - len(_val_ics) - _reused_chunks
                print(f"ML滚动训练完成: {len(chunk_starts)} chunks, 有效={len(_val_ics)} "
                      f"复用前模型={_reused_chunks} 跳过={_true_skip}, "
                      f"avg_IC={_ml_val_ic:.4f}, preds={_ml_total_preds:,}")
                if _val_ics:
                    print(f"  IC范围: [{min(_val_ics):.4f}, {max(_val_ics):.4f}], "
                          f"中位数={sorted(_val_ics)[len(_val_ics)//2]:.4f}")
                if _val_ics:
                    print(f"ML模型已保存: {_ml_model_path}")

        except ImportError as e:
            print(f"[ML] ImportError: {e}")
            import traceback; traceback.print_exc()
        except Exception as e:
            print(f"[ML] 训练失败: {e}")
            import traceback; traceback.print_exc()

    # 准备信号生成用的参数
    global use_dynamic, precomputed_cache
    use_dynamic = factor_mode != 'fixed'
    stock_codes = [name for name in stock_file_map.keys() if name != "sh000001"]

    # 创建带动态因子的 SignalEngine + 预计算因子选择（如果启用）
    main_engine = None
    _rank_parquet_path = None  # 预定义，避免 use_dynamic=False 时 UnboundLocalError
    if use_dynamic:
        main_engine = SignalEngine()
        main_engine.set_factor_data(factor_df)
        main_engine.set_industry_mapping(industry_codes)
        main_engine.set_fundamental_data(fundamental_data)
        if _ml_model_path is not None and _ml_preds:
            from core.ml_predictor import MLFactorPredictor
            ml_predictor = MLFactorPredictor(config.config)
            ml_predictor.load_model(_ml_model_path)
            main_engine.set_ml_predictor(ml_predictor)
            main_engine.set_ml_predictions(_ml_preds)
        print(f"主引擎已设置动态因子数据")

        # === 截面排名缓存：从 factor_df 提取 _rank 列，写入 parquet 供 worker 按需读取 ===
        _rank_cols = [c for c in factor_df.columns if c.endswith('_rank')]
        if _rank_cols:
            strategy_dir = os.path.dirname(os.path.abspath(__file__))
            _rank_parquet_path = os.path.join(strategy_dir, 'rolling_validation_results',
                                              'factor_ranks_worker.parquet')
            # 性能(2026-10-09): 按(code,date)排序+小组行组 → worker每股读排名从
            # 半个表(83万行解压)降到~1-2个行组(2-4万行)
            import pyarrow as pa
            import pyarrow.parquet as pq
            _rank_df = factor_df[['code', 'date'] + _rank_cols].sort_values(
                ['code', 'date'], kind='mergesort')
            _rank_tbl = pa.Table.from_pandas(_rank_df, preserve_index=False)
            pq.write_table(_rank_tbl, _rank_parquet_path, row_group_size=20000, compression='zstd')
            del _rank_df, _rank_tbl
            main_engine.set_rank_parquet(_rank_parquet_path)
            strategy.signal_engine.set_rank_parquet(_rank_parquet_path)
            print(f"截面排名缓存: {len(_rank_cols)} 个排名因子 → {_rank_parquet_path}")
            # 释放 _rank 列内存 (~920MB)，已写入 parquet 供 worker 按需读取
            factor_df.drop(columns=_rank_cols, inplace=True)

        # 转换 float64 → float32 节省 ~1.1GB 内存 (IC/ML 均兼容 float32)
        _f64_cols = [c for c in factor_df.columns if factor_df[c].dtype == 'float64']
        if _f64_cols:
            factor_df[_f64_cols] = factor_df[_f64_cols].astype('float32')

        # 初始化因子库 (持久化IC评估 + 时变质量追踪)
        from core.factor_library import create_factor_library
        _factor_lib = create_factor_library()
        main_engine.dynamic_factor_selector.factor_library = _factor_lib

        # 预计算所有日期的因子选择（避免多进程中重复计算）
        print("预计算因子选择...")
        main_engine.dynamic_factor_selector.precompute_all_factor_selections(
            progress_callback=lambda curr, total: print(f"\r因子选择进度: {curr}/{total}", end="", flush=True),
            num_workers=NUM_WORKERS  # 内存已通过 _malloc_trim 控制，恢复并行
        )
        print(f"\n因子选择预计算完成，共 {len(main_engine.dynamic_factor_selector._factor_cache)} 个日期")

        # 提取预计算的缓存传递给workers
        precomputed_cache = main_engine.dynamic_factor_selector._factor_cache
        precomputed_all_dates = main_engine.dynamic_factor_selector._all_dates_cache

        # === 释放 factor_df（预计算完成，不再需要原始因子DataFrame） ===
        strategy.signal_engine.dynamic_factor_selector.factor_df = None
        main_engine.dynamic_factor_selector.factor_df = None
        main_engine.dynamic_factor_selector.industry_codes = {}
        del factor_df
        factor_df = None
        import gc
        gc.collect()
        gc.collect()
        _malloc_trim()  # 强制归还 freed pages 给 OS, 降低后续 fork 时的 RSS
        print("已释放 factor_df 内存（已归还 OS）")
    else:
        precomputed_cache = None
        precomputed_all_dates = None
        # 内存(2026-10-09): fixed模式(生产)ML段结束后 factor_df/_ml_df 零引用 —
        # 原释放只在dynamic分支, 主进程持~2.4GB进入信号段, 8 workers内存预算不足。
        # _ml_df在ML异常早退时可能未定义 → dir()守卫
        if '_ml_df' in dir():
            del _ml_df
        if 'factor_df' in dir():
            del factor_df
            factor_df = None
        import gc
        gc.collect()
        gc.collect()
        _malloc_trim()  # 归还OS, 降低后续spawn worker时的内存压力
        print("已释放 factor_df/_ml_df 内存（fixed模式）")

    # ===================================================================
    # Phase B: 创建 Backtrader datafeeds（从磁盘重新加载，避免与 factor_df 共存）
    # factor_df 已释放，此时内存峰值最低
    # ===================================================================
    price_cols = ['open', 'high', 'low', 'close']
    for name in tqdm(stock_codes, desc="preparing datafeeds"):
        filepath = stock_file_map.get(name)
        if not filepath:
            continue
        data = pd.read_csv(filepath, parse_dates=['datetime'], dtype=_DTYPE_MAP)
        if FROMDATE:
            data = data[data['datetime'] >= FROMDATE]
        if TODATE:
            data = data[data['datetime'] <= TODATE]
        if 'datetime' in data.columns:
            data = data.set_index('datetime')
        data = data.reindex(calendar_index)
        data[price_cols] = data[price_cols].ffill()
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)
            if data['volume'].dtype != 'float64':
                data['volume'] = data['volume'].astype('float64')
        datafeed = bt.feeds.PandasData(dataname=data[price_cols + ['volume']])
        cerebro.adddata(datafeed, name=name)

    # 准备参数：传文件路径给worker（而非dict），避免内存翻倍
    stock_items = [
        (name, stock_file_map[name])
        for name in stock_codes
    ]

    # 增量写入信号CSV（而非内存中累积 ~1.3GB 的 all_signals 列表）
    # 原子写: 先写.tmp再os.replace — 中途被杀不损坏既有CSV
    # (2026-09-06教训: H5-a运行在信号生成中被杀, 旧CSV随os.remove+open('w')丢失,
    #  被迫78min重生成。原子写后旧CSV保留到新CSV完整落盘为止)
    strategy_dir = os.path.dirname(os.path.abspath(__file__))
    signals_output_path = os.path.join(strategy_dir, 'rolling_validation_results', 'backtest_signals.csv')
    os.makedirs(os.path.dirname(signals_output_path), exist_ok=True)
    signal_csv = open(signals_output_path + '.tmp', 'w', encoding='utf-8')
    signal_csv.write('code,date,buy,sell,score,pre_discount_score,factor_value,factor_name,industry,factor_quality,'
                     'chan_divergence_type,chan_divergence_strength,chan_structure_score,'
                     'chan_buy_point,chan_sell_point,bi_buy_point,signal_level,trend_type,'
                     'chan_pivot_zg,chan_pivot_zd,mom_60d,dist_ma60,max_dd_20d,vol_regime,'
                     'mtf_discount_factor,mtf_alignment_score,avg_trend_strength,'
                     'risk_vol,daily_return,volume_ratio,stroke_phase,exhaustion_risk,'
                     'gap_breakout_confirm,vol_opening_confirm,vol_opening_strength,'
                     'bom_quality_score,gate_quality,profit_declining,ma_trend_up,'
                     'ml_score,adjusted_score\n')
    signal_count = [0]  # 用list实现闭包写入计数

    # 多进程并行生成信号
    print(f"多进程生成信号 ({NUM_WORKERS} workers)...")
    _worker_dyn_fail_agg = {k: 0 for k in ['total_calls', 'no_code_or_date', 'no_industry',
                                             'no_cache_data', 'no_dates', 'lookup_fail',
                                             'low_quality', 'no_factor_score']}

    # 动态因子统计
    global dynamic_factor_stats
    dynamic_factor_stats = {'hit': 0, 'miss': 0, 'factor_names': {}}

    # === 保存 ML 预测到 parquet（按code索引, worker按需读取, 避免69MB/进程pickle OOM）===
    # 性能(2026-10-09): 按(code,date)排序 + row_group_size=20000 → pyarrow filter
    # 可剪枝到单行组 (旧布局4个行组全横跨全代码范围, 每股读ML预测全扫322万行)
    _ml_preds_path = None
    if _ml_codes_arr is not None and len(_ml_codes_arr) > 0:
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        _ml_preds_path = os.path.join(strategy_dir, 'rolling_validation_results', 'ml_preds_worker.parquet')
        import pyarrow as pa
        import pyarrow.parquet as pq
        ml_preds_df = pd.DataFrame(
            {'code': _ml_codes_arr, 'date': _ml_dates_arr, 'ml_pred': _ml_preds_arr})
        ml_preds_df = ml_preds_df.sort_values(['code', 'date'], kind='mergesort')
        _tbl = pa.Table.from_pandas(ml_preds_df, preserve_index=False)
        pq.write_table(_tbl, _ml_preds_path, row_group_size=20000, compression='zstd')
        del ml_preds_df, _tbl

    # 预初始化另类数据缓存（主进程单次执行，避免8个worker同时下载）
    try:
        from core.alternative_data import get_provider
        alt = get_provider()
        alt._ensure_dragon_tiger_history()
        alt.get_northbound_signal(pd.Timestamp('2024-01-02').date())
        alt.get_margin_signal(pd.Timestamp('2024-01-02').date())
    except Exception:
        pass

    # 性能(2026-10-09): 季度因子配置YAML(4.3MB/23季度)在主进程预解析一次经initargs传入,
    # worker不再各自解析 (~15s/worker首股税消除, 且避免8 worker并发YAML解析CPU峰)
    _quarter_configs = {}
    try:
        import core.signal_engine as _se_mod
        for _qid in list(_se_mod._load_quarter_index().keys()):
            _se_mod._load_quarter_config(_qid)
        _quarter_configs = _se_mod._QUARTER_CONFIG_CACHE
    except Exception:
        pass

    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(
        processes=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(FUNDAMENTAL_PATH, stock_codes, use_dynamic, industry_codes,
                  precomputed_cache, precomputed_all_dates, regime_df,
                  _ml_model_path, _ml_preds_path, _rank_parquet_path, _quarter_configs)
    )
    _sig_iter = pool.imap_unordered(_generate_stock_signal_worker, stock_items, chunksize=50)
    # 聚合 worker 因子选择统计
    _worker_stats_agg = {}
    _worker_bom_agg = {'total': 0, 'hit': 0, 'miss': 0, 'moat': 0, 'sum_score': 0.0,
                       'n_unique': 0, 'n_moat_codes': 0}
    for result in _sig_iter:
        code, store_data, dyn_fail, diag_data, stats_data, bom_data = (
            result if len(result) >= 6 else
            (result[0], result[1], result[2],
             result[3] if len(result) > 3 else {},
             result[4] if len(result) > 4 else {},
             result[5] if len(result) > 5 else {}))
        for k, v in dyn_fail.items():
            if k in _worker_dyn_fail_agg:
                _worker_dyn_fail_agg[k] += v
        # 合并 worker 因子选择统计（跳过列表类型字段）
        for k, v in stats_data.items():
            if isinstance(v, (int, float)):
                _worker_stats_agg[k] = _worker_stats_agg.get(k, 0) + v
        # 聚合 worker BOM 统计
        if bom_data:
            for k in ('total', 'hit', 'miss', 'moat', 'n_unique', 'n_moat_codes'):
                _worker_bom_agg[k] = _worker_bom_agg.get(k, 0) + bom_data.get(k, 0)
            _worker_bom_agg['sum_score'] = _worker_bom_agg.get('sum_score', 0.0) + bom_data.get('sum_score', 0.0)
        # 合并 worker 诊断数据到主进程
        if diag_data:
            try:
                from analysis.backtest_diagnostics import get_diagnostics
                main_diag = get_diagnostics()
                m = main_diag.metrics
                for k in ['factor_selection', 'buy_by_buy_point', 'buy_by_factor_family']:
                    if k in diag_data:
                        for kk, vv in diag_data[k].items():
                            m[k][kk] = m[k].get(kk, 0) + vv
                m['buy_signals'] = m.get('buy_signals', 0) + diag_data.get('buy_signals', 0)
                m['hard_rejects'] = m.get('hard_rejects', 0) + diag_data.get('hard_rejects', 0)
                m['gate_quality_samples'].extend(diag_data.get('gate_quality_samples', []))
                m['alt_northbound_hits'] = m.get('alt_northbound_hits', 0) + diag_data.get('alt_nb', 0)
                m['alt_margin_hits'] = m.get('alt_margin_hits', 0) + diag_data.get('alt_mg', 0)
                m['alt_dragon_tiger_hits'] = m.get('alt_dragon_tiger_hits', 0) + diag_data.get('alt_dt', 0)
                # A.6-1b(2026-09-22): 主进程聚合worker回传的季度诊断计数
                if 'quarter_diag' in diag_data:
                    try:
                        from core import signal_engine as _se_mod
                        for _k, _v in diag_data['quarter_diag'].items():
                            if _k in _se_mod._QUARTER_DIAG:
                                _se_mod._QUARTER_DIAG[_k] += int(_v)
                    except Exception:
                        pass
            except Exception as e:
                import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()
        # 增量写入CSV（不在内存中累积 Signal 对象）
        for (c, date), sig in store_data.items():
            if hasattr(date, 'date'):
                date = date.date()
            row = (
                f'{c},{date},'
                f'{sig.buy},{sig.sell},{sig.score},{sig.pre_discount_score},'
                f'{sig.factor_value},{sig.factor_name},{sig.industry},'
                f'{getattr(sig, "factor_quality", 0.0)},'
                f'{getattr(sig, "chan_divergence_type", "")},'
                f'{getattr(sig, "chan_divergence_strength", 0.0)},'
                f'{getattr(sig, "chan_structure_score", 0.0)},'
                f'{getattr(sig, "chan_buy_point", 0)},'
                f'{getattr(sig, "chan_sell_point", 0)},'
                f'{getattr(sig, "bi_buy_point", 0)},'
                f'{getattr(sig, "signal_level", 0)},'
                f'{getattr(sig, "trend_type", 0)},'
                f'{getattr(sig, "chan_pivot_zg", float("nan"))},'
                f'{getattr(sig, "chan_pivot_zd", float("nan"))},'
                f'{getattr(sig, "mom_60d", 0.0)},'
                f'{getattr(sig, "dist_ma60", 0.0)},'
                f'{getattr(sig, "max_dd_20d", 0.0)},'
                f'{getattr(sig, "vol_regime", 1.0)},'
                f'{getattr(sig, "mtf_discount_factor", 1.0)},'
                f'{getattr(sig, "mtf_alignment_score", 0.0)},'
                f'{(getattr(sig, "weekly_trend_strength", 0.0) + getattr(sig, "monthly_trend_strength", 0.0)) / 2},'
                f'{getattr(sig, "risk_vol", 0.0)},'
                f'{getattr(sig, "daily_return", 0.0)},'
                f'{getattr(sig, "volume_ratio", 0.0)},'
                f'{getattr(sig, "stroke_phase", 0.0)},'
                f'{getattr(sig, "exhaustion_risk", 0.0)},'
                f'{getattr(sig, "gap_breakout_confirm", 0.0)},'
                f'{getattr(sig, "vol_opening_confirm", 0.0)},'
                f'{getattr(sig, "vol_opening_strength", 0.0)},'
                f'{getattr(sig, "bom_quality_score", 0.3)},'
                f'{getattr(sig, "_gate_quality", 0.5)},'
                f'{getattr(sig, "profit_declining", False)},'
                f'{getattr(sig, "ma_trend_up", False)},'
                f'{getattr(sig, "ml_score", 0.0)},'
                f'{getattr(sig, "adjusted_score", 0.0)}\n'
            )
            signal_csv.write(row)
            signal_count[0] += 1
            # 动态因子统计
            if sig.factor_name and sig.factor_name.startswith('DYN_'):
                dynamic_factor_stats['hit'] += 1
                fn = sig.factor_name.split('_')[1] if '_' in sig.factor_name else sig.factor_name
                dynamic_factor_stats['factor_names'][fn] = dynamic_factor_stats['factor_names'].get(fn, 0) + 1
            else:
                dynamic_factor_stats['miss'] += 1
        # 释放该股票返回的 dict（Signal 已写入 CSV，不再需要）
        store_data.clear()

    pool.close()
    pool.join()
    for k, v in _worker_dyn_fail_agg.items():
        strategy.signal_engine._dyn_fail[k] = v

    signal_csv.close()
    os.replace(signals_output_path + '.tmp', signals_output_path)
    print(f"信号数据已保存: {signal_count[0]} 条 -> {signals_output_path}")

    # === 信号质量诊断 ===
    try:
        _sig_df = pd.read_csv(signals_output_path)
        if len(_sig_df) > 0:
            _buys = _sig_df[_sig_df['signal'] == 'buy']
            _sells = _sig_df[_sig_df['signal'] == 'sell']
            print(f"\n[信号诊断] 总计={len(_sig_df):,} 买入={len(_buys):,} 卖出={len(_sells):,}")
            if len(_buys) > 0:
                _cols = [c for c in ['score', 'confidence', 'factor_value'] if c in _buys.columns]
                for _c in _cols:
                    _vals = _buys[_c].dropna()
                    if len(_vals) > 0:
                        print(f"  买入{_c}: avg={_vals.mean():.4f} median={_vals.median():.4f} "
                              f"[{_vals.min():.4f}, {_vals.max():.4f}]")
            # 按年统计
            if 'date' in _sig_df.columns:
                _sig_df['year'] = pd.to_datetime(_sig_df['date']).dt.year
                for _yr in sorted(_sig_df['year'].unique()):
                    _yr_buy = len(_sig_df[(_sig_df['year'] == _yr) & (_sig_df['signal'] == 'buy')])
                    print(f"  {_yr}: 买入信号={_yr_buy}")
    except Exception as _e:
        pass

    # === 从CSV加载信号到 DataFrame-backed SignalStore（节省 ~800MB 内存） ===
    strategy.signal_store.finalize(signals_output_path, membership_map=membership_map)

    # 释放不再需要的大对象
    del stock_items
    import gc
    gc.collect()

    # 打印季度因子匹配诊断
    try:
        from core.signal_engine import _print_quarter_diag
        _print_quarter_diag()
    except Exception:
        pass

    # 打印FactorLibrary来源统计
    try:
        if hasattr(strategy, 'signal_engine'):
            _lib = getattr(strategy.signal_engine, 'factor_library', None) or \
                   getattr(getattr(strategy.signal_engine, 'dynamic_factor_selector', None), 'factor_library', None)
            if _lib and hasattr(_lib, 'print_diag'):
                _lib.print_diag()
    except Exception:
        pass

    # 打印动态因子统计
    total = dynamic_factor_stats['hit'] + dynamic_factor_stats['miss']
    if total > 0:
        hit_rate = dynamic_factor_stats['hit'] / total * 100
        print(f"\n=== 动态因子统计 ===")
        print(f"动态因子命中: {dynamic_factor_stats['hit']:,} / {total:,} ({hit_rate:.1f}%)")
        print(f"非动态因子: {dynamic_factor_stats['miss']:,}")
        if dynamic_factor_stats['factor_names']:
            print("行业因子分布:")
            for fn, cnt in sorted(dynamic_factor_stats['factor_names'].items(), key=lambda x: -x[1])[:10]:
                print(f"  {fn}: {cnt:,}")

    # 更新策略的基本面数据加载范围（限制为实际回测池股票，节省查询内存）
    if hasattr(strategy, 'portfolio') and hasattr(strategy.portfolio, 'fundamental_data'):
        from core.fundamental import FundamentalData
        fundamental_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'stock_data', 'fundamental_data'
        )
        strategy.portfolio.fundamental_data = FundamentalData(fundamental_path + '/', stock_codes=stock_codes)
        print(f"基本面数据已限制为 {len(stock_codes)} 只股票")

    # 释放 stock_file_map + 最终清理
    stock_file_map.clear()
    del stock_file_map
    import gc
    gc.collect()
    _malloc_trim()
    print("已释放所有缓存内存")

    # 注入 worker 聚合的因子选择统计到主引擎（保留 ic_values 列表）
    if _worker_stats_agg:
        for k, v in _worker_stats_agg.items():
            if k != 'ic_values':
                strategy.signal_engine._stats[k] = v

    # 注入 worker 聚合的 BOM 统计到主引擎
    if _worker_bom_agg:
        bd = strategy.signal_engine._bom_diag
        bd['total'] = _worker_bom_agg.get('total', 0)
        bd['hit'] = _worker_bom_agg.get('hit', 0)
        bd['miss'] = _worker_bom_agg.get('miss', 0)
        bd['moat'] = _worker_bom_agg.get('moat', 0)
        bd['sum_score'] = _worker_bom_agg.get('sum_score', 0.0)
        bd['_unique_codes'] = set(range(_worker_bom_agg.get('n_unique', 0)))  # placeholder
        bd['_moat_codes'] = set(range(_worker_bom_agg.get('n_moat_codes', 0)))

    # 打印因子选择统计
    strategy.signal_engine.print_factor_stats()
    strategy.signal_engine.print_bom_stats()
    strategy.signal_engine.print_score_diagnostics()

    # === 回测诊断报告 ===
    try:
        from analysis.backtest_diagnostics import get_diagnostics
        diag = get_diagnostics()
        if _ml_total_preds > 0:
            diag.record_ml(total=_ml_total_preds, active=_ml_total_preds, val_ic=_ml_val_ic)
        diag.print_report()
        diag.save()
    except Exception as e:
        print(f"[诊断] 报告生成失败: {e}")

class BacktraderExecution(bt.Strategy):
    params = dict(
        real_strategy=None,
    )

    # 数据质量过滤参数
    MIN_PRICE = 2.0  # 最低价格限制（前复权后），避免低价股异常
    MIN_VOLUME = 100  # 最低成交量，过滤停牌股

    def __init__(self):
        self.universe = [d._name for d in self.datas]
        self.count = 0
        self.current_rebalance_period = REBALANCE_DAYS  # 动态周期，默认20天
        self.orders_list = defaultdict(list)
        self.last_date = None
        self.cost = defaultdict(list)
        self.portfolio_selections = []  # 记录每期选股结果
        self._prev_total_value = None  # 用于计算每日收益率
        # ST/涨跌停配置
        st_config = config.get('stock_pool', {})
        self._filter_st = st_config.get('filter_st', True)  # 默认过滤ST
        self._st_codes = getattr(self.p.real_strategy, 'st_codes', set()) if self.p.real_strategy else set()
        # 涨跌停数据
        self._limit_data = _LIMIT_DATA  # 引用全局预计算数据

        # === 成交率追踪 ===
        self._fill_stats = {
            'buy_attempted': 0, 'buy_filled': 0, 'buy_partial': 0, 'buy_rejected': 0,
            'sell_attempted': 0, 'sell_filled': 0, 'sell_partial': 0, 'sell_rejected': 0,
            'buy_limit_up_skip': 0, 'sell_limit_down_skip': 0,
            'sell_tplus1_blocked': 0, 'buy_cash_insufficient': 0,
        }

        # === T+1结算 ===
        cm_config = config.get('cost_model', {})
        self._tplus1 = cm_config.get('t_plus_1_enabled', True)
        self._today_buys = set()

        # === 冲击成本 ===
        self._impact_enabled = cm_config.get('impact_cost_enabled', False)
        self._impact_base = cm_config.get('impact_cost_base', 0.0003)
        self._impact_exp = cm_config.get('impact_cost_exponent', 0.5)
        self._impact_min = cm_config.get('min_impact_cost', 0.00005)
        self._adv_lookback = cm_config.get('adv_lookback', 20)
        self._adv_cache = {}  # 由全局_ADV_CACHE填充

    def _is_tradable(self, d):
        """检查股票是否可交易（非停牌、价格正常、非ST/科创板）"""
        price = d.close[0]
        volume = d.volume[0] if hasattr(d, 'volume') else 1

        # 1. 价格必须为正
        if price is None or math.isnan(price) or price <= 0:
            return False

        # 2. 价格不能太低（前复权后）- 避免低价股异常放大收益
        if price < self.MIN_PRICE:
            return False

        # 3. 成交量必须大于0（非停牌）
        if volume is None or math.isnan(volume) or volume < self.MIN_VOLUME:
            return False

        # 4. ST股票过滤（默认从strategy获取，可通过配置关闭）
        if self._filter_st and d._name in self._st_codes:
            return False

        return True

    def _is_limit_up(self, code, date):
        """检查股票当日是否涨停（无法买入）"""
        if hasattr(date, 'date'):
            date = date.date()
        return _LIMIT_DATA.get((code, date)) == 'up'

    def _is_limit_down(self, code, date):
        """检查股票当日是否跌停（无法卖出）"""
        if hasattr(date, 'date'):
            date = date.date()
        return _LIMIT_DATA.get((code, date)) == 'down'

    def next(self):
        self._today_buys = set()  # T+1: 每日重置当日买入集合
        if self.last_date is not None and self.last_date in self.orders_list:
            for order in self.orders_list[self.last_date]:
                self.cancel(order)
            del self.orders_list[self.last_date]
        self.count += 1
        date = self.datas[0].datetime.date(0)
        self.last_date = date

        prices = {}
        tradable_universe = []  # 可交易的股票池
        for d in self.datas:
            if not self._is_tradable(d):
                continue
            price = d.close[0]
            prices[d._name] = price
            tradable_universe.append(d._name)

        current_positions = {
            d._name: self.getposition(d).size * prices[d._name]
            for d in self.datas
            if d._name in prices and self.getposition(d).size != 0
        }

        rebalance = False
        if self.count >= self.current_rebalance_period:
            self.count = 1
            rebalance = True
        target = self.p.real_strategy.generate_positions(
            date=date,
            universe=tradable_universe,
            current_positions=current_positions,
            cash=self.broker.getcash(),
            prices=prices,
            cost=self.cost,
        )

        # 记录选股结果
        if rebalance:
            selection = self.p.real_strategy.portfolio.last_selection
            for s in selection:
                self.portfolio_selections.append({
                    'date': date,
                    'code': s['code'],
                    'score': s['score'],
                    'weight': s['weight'],
                    'industry': s.get('industry', ''),
                })

        active_codes = set(target.keys()) | set(current_positions.keys())
        for d in self.datas:
            code = d._name
            if code not in active_codes:
                continue

            # 再次检查可交易性（买入时必须可交易，卖出可以放宽）
            if not self._is_tradable(d) and code not in current_positions:
                continue

            price = d.close[0]
            if price is None or math.isnan(price) or price <= 0:
                continue

            pos = self.getposition(d)
            current_value = pos.size * price
            target_value = target.get(code, 0.0)

            diff_value = target_value - current_value

            # 忽略极小调整
            if abs(diff_value) < price * 100:
                continue

            raw = diff_value / price / 100
            # A 股最小 100 股
            size = max(int(raw), 1) * 100 if raw > 0 else min(int(raw), -1) * 100

            if size > 0:
                # 涨停检查：无法买入
                if self._is_limit_up(code, date):
                    self._fill_stats['buy_attempted'] += 1
                    self._fill_stats['buy_limit_up_skip'] += 1
                    continue
                # 冲击成本：从现金中预留
                impact = self._estimate_impact(code, size, price)
                impact_value = impact * size * price
                effective_cash = self.broker.getcash() - impact_value
                max_affordable = int(max(0, effective_cash) / price / 100) * 100
                if max_affordable < 100:
                    self._fill_stats['buy_attempted'] += 1
                    self._fill_stats['buy_cash_insufficient'] += 1
                    continue
                size = min(size, max_affordable)
                self._fill_stats['buy_attempted'] += 1
                if size < abs(raw) * 100 * 0.5:
                    self._fill_stats['buy_partial'] += 1
                order = self.buy(data=d, size=size)
                self.orders_list[date].append(order)
                self._today_buys.add(code)
            elif size < 0:
                # T+1: 当日买入不可卖出
                if self._tplus1 and code in self._today_buys:
                    self._fill_stats['sell_attempted'] += 1
                    self._fill_stats['sell_tplus1_blocked'] += 1
                    continue
                # 跌停检查：无法卖出
                if self._is_limit_down(code, date):
                    self._fill_stats['sell_attempted'] += 1
                    self._fill_stats['sell_limit_down_skip'] += 1
                    continue
                self._fill_stats['sell_attempted'] += 1
                order = self.sell(data=d, size=abs(size))
                self.orders_list[date].append(order)

        # 计算并记录每日收益率（供组合层波动率控制使用）
        total_value = self.broker.getvalue()
        if self._prev_total_value is not None and self._prev_total_value > 0:
            daily_return = (total_value - self._prev_total_value) / self._prev_total_value
            self.p.real_strategy.portfolio.update_returns(daily_return)
        self._prev_total_value = total_value

    def _estimate_impact(self, code: str, size: int, price: float) -> float:
        """估算冲击成本（交易量/日均量的平方根缩放）

        注意：向量化路径已替代 BT 路径，此方法暂未使用。
        如需启用 BT 路径，需先填充 self._adv_cache（从 datafeeds 预计算日均量）。
        """
        if not self._impact_enabled:
            return 0.0
        adv = self._adv_cache.get(code, 0)
        if adv <= 0:
            return 0.0
        participation = (abs(size) * price) / max(adv * price, 1)
        impact = self._impact_base * (participation ** self._impact_exp)
        return max(impact, self._impact_min)

    def print_fill_stats(self):
        """回测结束时输出成交率统计"""
        s = self._fill_stats
        print("\n" + "=" * 60)
        print("成交率统计 (Fill Rate)")
        print("=" * 60)
        ba = max(s['buy_attempted'], 1)
        sa = max(s['sell_attempted'], 1)
        print("买入: {}/{} = {:.1f}%  (涨停跳过={}, 现金不足={}, 部分成交={})".format(
            s['buy_filled'], s['buy_attempted'], s['buy_filled']/ba*100,
            s['buy_limit_up_skip'], s['buy_cash_insufficient'], s['buy_partial']))
        print("卖出: {}/{} = {:.1f}%  (跌停跳过={}, T+1拦截={})".format(
            s['sell_filled'], s['sell_attempted'], s['sell_filled']/sa*100,
            s['sell_limit_down_skip'], s['sell_tplus1_blocked']))

    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已经处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.status == order.Completed:
                if order.isbuy():
                    self._fill_stats['buy_filled'] += 1
                if order.isbuy():
                    cost = self.cost[order.data._name]
                    if len(cost) == 0:
                        cost = [0, 0.0]
                    cost[1] = (cost[0] * cost[1] + order.executed.size * order.executed.price) / (cost[0] + order.executed.size)
                    cost[0] = cost[0] + order.executed.size
                    self.cost[order.data._name] = cost
                else:
                    cost = self.cost[order.data._name]
                    cost[0] = cost[0] + order.executed.size
                    if cost[0] == 0:
                        del self.cost[order.data._name]
                    else:
                        self.cost[order.data._name] = cost
            date = datetime.date.fromordinal(int(order.executed.dt))
            if order.isbuy():
                print(f'BUY EXECUTED, date {date}, ref: {order.ref}，Price: {order.executed.price}, '
                      f'Cost: {order.executed.value}, Comm {order.executed.comm}, Size: {order.executed.size}, Stock: {order.data._name}')
            else: # Sell
                print(f'SELL EXECUTED, date {date}, ref: {order.ref}, Price: {order.executed.price}, '
                      f'Cost: {order.executed.value}, Comm {order.executed.comm}, Size: {order.executed.size}, Stock: {order.data._name}')

def _vectorized_backtest(strategy, fundamental_data, fromdate, todate, initial_cash=100000):
    """向量化回测 — 替换 backtrader 逐 bar 循环。策略逻辑完全不变。"""
    import numpy as np
    from tqdm import tqdm

    # 1. 构建价格矩阵
    stock_codes = sorted([f.replace('_qfq.csv', '') for f in os.listdir(DATA_PATH)
                         if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv' and not f.startswith('._')])
    # 使用真实A股交易日历（指数数据），避免bdate_range包含节假日
    _idx_fp = os.path.join(DATA_PATH, 'sh000001_qfq.csv')
    _idx_df = pd.read_csv(_idx_fp, parse_dates=['datetime'])
    _idx_df = _idx_df[(_idx_df['datetime'] >= pd.Timestamp(fromdate)) &
                      (_idx_df['datetime'] <= pd.Timestamp(todate))]
    calendar = pd.DatetimeIndex(sorted(_idx_df['datetime'].unique()))
    n_dates = len(calendar)
    _STAMP_ARR = np.where(calendar.values < np.datetime64('2023-08-28'), 0.001, 0.0005).astype(np.float64)  # 印花税分档: 2023-08-28前万10后万5 (9/22成本审计)
    code_to_idx = {c: i for i, c in enumerate(stock_codes)}
    idx_to_code = {i: c for i, c in enumerate(stock_codes)}

    print(f"构建价格矩阵: {n_dates}天 × {len(stock_codes)}股 ...")
    close_px = np.full((n_dates, len(stock_codes)), np.nan, dtype=np.float32)
    open_px = np.full_like(close_px, np.nan)  # B批修复(B3): T+1开盘成交价
    low_px = np.full_like(close_px, np.nan)  # 分段建仓(2026-10-10): 限价单触发用当日最低价
    volume_m = np.zeros_like(close_px)
    tradable = np.zeros_like(close_px, dtype=bool)

    for j, code in enumerate(tqdm(stock_codes, desc="loading")):
        fp = os.path.join(DATA_PATH, f'{code}_qfq.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, usecols=['datetime', 'open', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df = df.set_index('datetime')
        df = df[df.index >= pd.Timestamp(fromdate)]
        df = df[df.index <= pd.Timestamp(todate)]
        df = df.reindex(calendar)
        close_px[:, j] = df['close'].ffill().values.astype(np.float32)
        open_px[:, j] = df['open'].ffill().values.astype(np.float32)
        low_px[:, j] = df['low'].ffill().values.astype(np.float32)
        volume_m[:, j] = df['volume'].fillna(0).values.astype(np.float32)
    del df

    # 预计算滚动ADV矩阵（20日回溯，shift(1)排除当日，无前视偏差）
    # volume单位为手, ×100转股数(与下单股数同口径, 否则participation高估100x)
    _vol_df = pd.DataFrame(volume_m * 100.0, index=calendar, columns=stock_codes)
    _adv_matrix = _vol_df.rolling(20, min_periods=5).mean().shift(1).values
    _adv_matrix = np.nan_to_num(_adv_matrix, nan=0.0)

    # 冲击成本参数
    cm_config = config.get('cost_model', {})
    impact_enabled = cm_config.get('impact_cost_enabled', True)
    impact_base = cm_config.get('impact_cost_base', 0.0003)
    impact_exp = cm_config.get('impact_cost_exponent', 0.5)
    impact_min = cm_config.get('min_impact_cost', 0.00005)

    # 滑点与涨跌停拦截 (诚实成本模型)
    slip_enabled = cm_config.get('slippage_enabled', True)
    slip_rate = float(config.get('backtest.slippage', 0.001))
    limit_block_enabled = cm_config.get('limit_fill_block_enabled', True)
    # 0b(2026-09-09): 跳空>9.5%再折让 — 原×1.03/×0.97产生超涨跌停价的"不可能
    # 成交价"(主板跳空+9.8%时买价=昨收×1.132>涨停×1.10; 卖侧对称低于跌停)
    gap_penalty_enabled = cm_config.get('gap_fill_penalty_enabled', True)
    # 分段建仓(实验, 2026-10-10): 新入场拆首段开盘成交+第二段限价单(坑底加仓),
    # 未成交余量在下个调仓日开盘回补(现金自动循环). 默认关=生产锚点不变.
    tranche_enabled = config.get('backtest', {}).get('tranche_enabled', False)
    tranche_ratio = float(config.get('backtest', {}).get('tranche_ratio', 0.5))
    tranche_discount = float(config.get('backtest', {}).get('tranche_discount', 0.05))

    def _impact(adv, size, price):
        if not impact_enabled or adv <= 0:
            return 0.0
        participation = abs(size) / max(adv, 1)
        return max(impact_base * (participation ** impact_exp), impact_min)

    # 涨跌停矩阵: 创业板30/科创板68=20%, 北交所4x/8x=30%, 其余主板10%
    # 前复权比例不受复权影响(除权日除外, 近似可接受); 触板阈值留0.3%容差
    _limit_rate_arr = np.array([
        0.20 if c[:2] in ('30', '68') else (0.30 if c[0] in ('4', '8') else 0.10)
        for c in stock_codes
    ])
    # 跳空罚阈值按板块(留0.5pp容差, 与a77d107信号层口径一致): 主板9.5%/20%板19.5%/北交所29.5%
    # #289(2026-10-12): 修复三处一刀切0.095 — 20%板gap 9.5%~19.5%为正常跳空(未触板), 罚无依据
    _gap_thr_arr = np.array([
        0.195 if c[:2] in ('30', '68') else (0.295 if c[0] in ('4', '8') else 0.095)
        for c in stock_codes
    ])
    _gap_dbg = os.environ.get('QUANT_GAP_DEBUG') == '1'  # 逐笔跳空罚日志(调试)
    _pct_chg = np.full_like(close_px, np.nan)
    _pct_chg[1:] = close_px[1:] / close_px[:-1] - 1.0
    _limit_up = _pct_chg >= (_limit_rate_arr[None, :] - 0.003)
    _limit_down = _pct_chg <= -(_limit_rate_arr[None, :] - 0.003)
    del _pct_chg

    # 2. 预计算可交易矩阵（含流动性过滤：日均成交额>500万）
    # volume单位为手, ×100换算真实成交额(元); 2026-08-22修复: 此前漏乘100导致阈值实际为5亿
    daily_value = close_px * volume_m * 100.0  # 每日成交额(元)
    # 20日滚动均值: 用 pandas rolling (沿 time axis=0)
    dv = pd.DataFrame(daily_value, index=calendar, columns=stock_codes)
    avg_daily_value = dv.rolling(20, min_periods=5).mean().shift(1).values
    tradable = (~np.isnan(close_px)) & (close_px > 2.0) & (volume_m > 100) & (avg_daily_value > 5e6)
    # ST 过滤: 预计算每只股票ST状态时间线后向量化匹配交易日
    # 2026-08-22性能优化: 原双重循环约250万次is_st调用(每次含过滤+排序)耗时~42min
    if fundamental_data is not None and hasattr(fundamental_data, 'get_st_timeline'):
        cal_str = np.array([d.strftime('%Y%m%d') for d in calendar])
        for j, code in enumerate(stock_codes):
            try:
                tl = fundamental_data.get_st_timeline(code)
                if tl is None:
                    continue
                _av, _stf = tl
                idx = np.searchsorted(_av, cal_str, side='right') - 1
                m = idx >= 0
                if not m.any():
                    continue
                col = np.zeros(n_dates, dtype=bool)
                col[m] = _stf[idx[m]]
                tradable[:, j] &= ~col
            except Exception as e:
                import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()

    # 3. 主循环
    print("运行回测 ...")
    cash = float(initial_cash)
    positions = np.zeros(len(stock_codes), dtype=np.int32)
    nav = np.full(n_dates, np.nan)
    daily_ret = np.full(n_dates - 1, np.nan)
    selections = []
    cost_tracker = {}
    _entry_dates = {}     # code -> 首次建仓日期
    _realized_trades = []  # 逐笔平仓审计

    def _log_realized(code, date, sell_px):
        """全仓退出时记录逐笔交易 (口径与历史trade_realized.csv一致: ret=exit_px/avg_cost-1)"""
        ct = cost_tracker.get(code)
        if ct is None:
            return
        entry_d = _entry_dates.get(code, date)
        _realized_trades.append({
            'entry_date': entry_d, 'exit_date': date, 'code': code,
            'hold_days': (date - entry_d).days,
            'ret': sell_px / max(ct[1], 1e-9) - 1.0,
            'avg_cost': ct[1], 'exit_px': sell_px,
        })
        _entry_dates.pop(code, None)

    # 成交统计
    _fill_stats = {'buy_attempted': 0, 'buy_filled': 0, 'buy_partial': 0,
                   'sell_attempted': 0, 'sell_filled': 0, 'sell_partial': 0,
                   'buy_limit_up_skip': 0, 'sell_limit_down_skip': 0,
                   'sell_tplus1_blocked': 0, 'buy_cash_insufficient': 0,
                   'gap_buy_penalized': 0, 'gap_sell_penalized': 0,
                   'tranche_filled': 0, 'tranche_cash_insufficient': 0}

    # T+1 结算：当日买入的股票不可卖出
    tplus1_enabled = config.get('cost_model', {}).get('t_plus_1_enabled', True)
    _today_buys = set()

    _pending = {}  # B批修复(B3): 前一日收盘目标 → 次日开盘成交
    _pending_limits = {}  # 分段建仓: code -> [限价, 余量股数, 建仓日bar idx]
    _tranche_plan = {}  # 分段建仓: 当日首段成交后待注册的限价单
    for i in tqdm(range(n_dates), desc="backtest"):
        date = calendar[i].date()
        px_today = close_px[i]
        ok = tradable[i]
        _today_buys.clear()  # T+1: 每日重置

        # 获取当日市场状态 (用于分市场统计)
        _regime_today = 0
        if strategy.index_data is not None:
            _idx_row = strategy.index_data[strategy.index_data['datetime'].dt.date == date]
            if not _idx_row.empty:
                _regime_today = int(_idx_row['regime'].values[0])

        # ===== 开盘执行前一日挂单 (B批修复B3: T+1开盘成交, 替代T日收盘成交) =====
        # 先卖后买: 卖在开盘释放现金供买; 末日无D+1挂单作废, 首日无挂单
        if i > 0 and _pending:
            open_today = open_px[i]
            prev_close = close_px[i - 1]

            # Step A: 卖出不在目标中的持仓 (全清)
            for j in range(len(stock_codes)):
                if positions[j] <= 0:
                    continue
                code = stock_codes[j]
                if code in _pending:
                    continue  # 仍在目标中, 交给Step B调整
                if not ok[j]:
                    continue
                if np.isnan(open_today[j]):
                    continue
                _fill_stats['sell_attempted'] += 1
                if tplus1_enabled and code in _today_buys:
                    _fill_stats['sell_tplus1_blocked'] += 1
                    continue
                if limit_block_enabled and _limit_down[i, j]:
                    _fill_stats['sell_limit_down_skip'] += 1
                    continue  # 跌停无法卖出, 挂单作废
                sell_px = float(open_today[j])
                if slip_enabled:
                    sell_px *= (1.0 - slip_rate)
                if gap_penalty_enabled and not np.isnan(prev_close[j]) and prev_close[j] > 0 and (sell_px / prev_close[j] - 1) < -_gap_thr_arr[j]:
                    if _gap_dbg:
                        print(f"[GAP] 卖全清 {code} {date} gap={sell_px/prev_close[j]-1:+.4f} thr=-{_gap_thr_arr[j]:.3f}", flush=True)
                    sell_px = sell_px * 0.97  # 跳空低开超板阈: 成交价再折让
                    _fill_stats['gap_sell_penalized'] += 1
                impact = _impact(_adv_matrix[i, j], positions[j], sell_px)
                cash += float(positions[j]) * sell_px * (1.0 - COMMISSION - _STAMP_ARR[i] - impact)
                _log_realized(code, date, sell_px)
                positions[j] = 0
                cost_tracker.pop(code, None)  # 清仓: 移除成本记录
                _fill_stats['sell_filled'] += 1

            # Step B: 减仓 (释放现金)
            for code, tv in _pending.items():
                j = code_to_idx.get(code)
                if j is None or not ok[j]:
                    continue
                if np.isnan(open_today[j]):
                    continue
                openx = float(open_today[j])
                target_shares = int(tv / openx / 100) * 100
                curr_shares = int(positions[j])
                diff = target_shares - curr_shares
                if diff <= -100:  # 减仓 → 释放现金
                    _fill_stats['sell_attempted'] += 1
                    if tplus1_enabled and code in _today_buys:
                        _fill_stats['sell_tplus1_blocked'] += 1
                        continue
                    if limit_block_enabled and _limit_down[i, j]:
                        _fill_stats['sell_limit_down_skip'] += 1
                        continue
                    sell_px = openx
                    if slip_enabled:
                        sell_px *= (1.0 - slip_rate)
                    if gap_penalty_enabled and not np.isnan(prev_close[j]) and prev_close[j] > 0 and (sell_px / prev_close[j] - 1) < -_gap_thr_arr[j]:
                        if _gap_dbg:
                            print(f"[GAP] 卖减仓 {code} {date} gap={sell_px/prev_close[j]-1:+.4f} thr=-{_gap_thr_arr[j]:.3f}", flush=True)
                        sell_px = sell_px * 0.97
                        _fill_stats['gap_sell_penalized'] += 1
                    impact = _impact(_adv_matrix[i, j], abs(diff), sell_px)
                    cash += abs(diff) * sell_px * (1.0 - COMMISSION - _STAMP_ARR[i] - impact)
                    positions[j] = target_shares
                    # 减仓: 减少成本追踪股数 (均价不变)
                    if code in cost_tracker:
                        cost_tracker[code][0] = max(cost_tracker[code][0] - abs(diff), 0)
                        if cost_tracker[code][0] <= 0:
                            _log_realized(code, date, sell_px)
                            cost_tracker.pop(code, None)
                    _fill_stats['sell_filled'] += 1

            # Step C: 加仓 — 先算总需资金, 按比例缩放确保不超现金
            _buy_plan = []
            _total_buy_cost = 0.0
            _tranche_plan = {}
            for code, tv in _pending.items():
                j = code_to_idx.get(code)
                if j is None or not ok[j]:
                    continue
                if np.isnan(open_today[j]):
                    continue
                if limit_block_enabled and _limit_up[i, j]:
                    _fill_stats['buy_limit_up_skip'] += 1
                    continue  # 涨停无法买入, 挂单作废
                buy_px = float(open_today[j])
                if slip_enabled:
                    buy_px *= (1.0 + slip_rate)
                if gap_penalty_enabled and not np.isnan(prev_close[j]) and prev_close[j] > 0 and (buy_px / prev_close[j] - 1) > _gap_thr_arr[j]:
                    if _gap_dbg:
                        print(f"[GAP] 买 {code} {date} gap={buy_px/prev_close[j]-1:+.4f} thr={_gap_thr_arr[j]:.3f}", flush=True)
                    buy_px = buy_px * 1.03  # 跳空高开超板阈: 成交价再折让
                    _fill_stats['gap_buy_penalized'] += 1
                target_shares = int(tv / buy_px / 100) * 100
                curr_shares = int(positions[j])
                diff = target_shares - curr_shares
                if diff >= 100:
                    # 分段建仓(实验): 开盘回补取代限价单(撤单, 防超买)
                    if tranche_enabled and code in _pending_limits:
                        _pending_limits.pop(code, None)
                    # 新入场拆段: 首段开盘, 余量挂限价单(需两段各>=1手)
                    if tranche_enabled and curr_shares == 0 and diff >= 200:
                        first = max(int(diff * tranche_ratio / 100) * 100, 100)
                        rem = diff - first
                        if rem >= 100:
                            _tranche_plan[code] = (buy_px * (1.0 - tranche_discount), rem)
                            diff = first
                            target_shares = curr_shares + first
                    impact = _impact(_adv_matrix[i, j], diff, buy_px)
                    cost = diff * buy_px * (1.0 + COMMISSION + impact)
                    _buy_plan.append((code, j, buy_px, target_shares, curr_shares, diff, impact, cost))
                    _total_buy_cost += cost

            # 现金不足时按比例缩放所有买入
            _buy_scale = 1.0
            if _total_buy_cost > 0 and _total_buy_cost > cash:
                _buy_scale = cash / _total_buy_cost
            _buy_plan.sort(key=lambda x: -x[5])  # sort by diff descending
            for (code, j, buy_px, target_shares, curr_shares, diff, impact, cost) in _buy_plan:
                _fill_stats['buy_attempted'] += 1
                scaled_diff = max(int(diff * _buy_scale / 100) * 100, 0)
                if scaled_diff >= 100:
                    scaled_cost = scaled_diff * buy_px * (1.0 + COMMISSION + impact)
                    if scaled_cost <= cash + 1e-6:
                        cash -= scaled_cost
                        positions[j] = curr_shares + scaled_diff
                        _fill_stats['buy_filled'] += 1
                        if curr_shares == 0:
                            _entry_dates[code] = date  # 首次建仓=成交日(开盘)
                        # 更新成本追踪: [总股数, 加权均价]
                        if code in cost_tracker:
                            old_s, old_avg = cost_tracker[code]
                            new_s = old_s + scaled_diff
                            cost_tracker[code] = [new_s, (old_s * old_avg + scaled_diff * buy_px) / max(new_s, 1)]
                        else:
                            cost_tracker[code] = [scaled_diff, buy_px]
                        if tranche_enabled and code in _tranche_plan:
                            # 首段成交 → 注册限价单(缩放削减的股数回到余量)
                            _lp, _rem = _tranche_plan[code]
                            _pending_limits[code] = [_lp, _rem + (diff - scaled_diff), i]
                        if tplus1_enabled:
                            _today_buys.add(code)
                    else:
                        _fill_stats['buy_cash_insufficient'] += 1
                else:
                    _fill_stats['buy_cash_insufficient'] += 1
            _pending = {}  # 挂单执行完毕

            # 分段建仓(实验): 第二段限价单检查 — 当日low触及限价 → min(开盘,限价)成交
            if tranche_enabled and _pending_limits:
                for code in list(_pending_limits.keys()):
                    _lp, _rem, _entry_i = _pending_limits[code]
                    j = code_to_idx.get(code)
                    if j is None or not ok[j] or positions[j] <= 0:
                        _pending_limits.pop(code, None)  # 已清仓/不可交易 → 撤单
                        continue
                    if np.isnan(open_px[i, j]) or np.isnan(low_px[i, j]):
                        continue
                    if limit_block_enabled and _limit_up[i, j]:
                        continue  # 涨停买不进, 挂单保留
                    if low_px[i, j] > _lp:
                        continue  # 未触及限价
                    fill_px = min(float(open_px[i, j]), _lp)
                    if slip_enabled:
                        fill_px *= (1.0 + slip_rate)
                    impact = _impact(_adv_matrix[i, j], _rem, fill_px)
                    unit = fill_px * (1.0 + COMMISSION + impact)
                    buy_shares = _rem
                    if buy_shares * unit > cash + 1e-6:
                        buy_shares = max(int(cash / unit / 100) * 100, 0)
                        if buy_shares < 100:
                            _fill_stats['tranche_cash_insufficient'] += 1
                            continue  # 现金不足, 挂单保留
                    cash -= buy_shares * unit
                    positions[j] += buy_shares
                    _fill_stats['tranche_filled'] += 1
                    if code in cost_tracker:
                        old_s, old_avg = cost_tracker[code]
                        new_s = old_s + buy_shares
                        cost_tracker[code] = [new_s, (old_s * old_avg + buy_shares * fill_px) / max(new_s, 1)]
                    else:
                        cost_tracker[code] = [buy_shares, fill_px]
                    if tplus1_enabled:
                        _today_buys.add(code)
                    _rem -= buy_shares
                    if _rem < 100:
                        _pending_limits.pop(code, None)
                    else:
                        _pending_limits[code][1] = _rem

        # 当前持仓市值 (开盘成交后, 按当日收盘价计)
        pos_value = float(np.dot(positions.astype(np.float64), np.nan_to_num(px_today, 0)))
        nav[i] = cash + pos_value
        if i > 0:
            daily_ret[i - 1] = (nav[i] - nav[i - 1]) / max(nav[i - 1], 1.0)
            # 波动率控制喂数 — A批修复(2026-08-30 review): 旧代码只在死的Backtrader路径调用,
            # 主回测 _daily_returns 恒空 → vol control 从未生效
            strategy.portfolio.update_returns(daily_ret[i - 1])
            if i >= 60:  # 市场状态检测需要60根bar
                plog.log_regime_execution(_regime_today, daily_ret[i - 1], nav[i])

        # 事件驱动：每天调用 generate_positions（止损检查 + 信号买入 + 补票）
        universe = [c for c in stock_codes if ok[code_to_idx[c]]]
        prices = {c: float(px_today[code_to_idx[c]]) for c in universe}
        cur_pos = {}
        for j in range(len(stock_codes)):
            if positions[j] > 0 and ok[j]:
                cur_pos[stock_codes[j]] = float(positions[j]) * float(px_today[j])
        # 分段建仓(实验): 限价单余量计入持仓市值, 防止下个调仓日开盘回补吃掉限价窗口
        if tranche_enabled and _pending_limits:
            for code, (_lp, _rem, _ei) in _pending_limits.items():
                j = code_to_idx.get(code)
                if j is not None and positions[j] > 0 and ok[j]:
                    cur_pos[code] = cur_pos.get(code, 0.0) + _rem * float(px_today[j])

        try:
            target = strategy.generate_positions(
                date=date, universe=universe, current_positions=cur_pos,
                cash=cash, prices=prices, cost=cost_tracker)
        except Exception as e:
            import traceback
            print(f" [选股异常] date={date}: {e}")
            traceback.print_exc()
            target = {}

        # 诊断(env门控, 默认零改动): 记录闸前量与组合层/引擎层权益差
        _dg = {'date': str(date)[:10], 'nav': float(nav[i]), 'cash': float(cash),
               'pos_value': float(pos_value), 'sum_cur_pos': float(sum(cur_pos.values())),
               'n_cur_pos': len(cur_pos), 'n_pos_all': int(np.sum(positions > 0)),
               'n_ok_pos': int(np.sum((positions > 0) & ok)),
               'n_sel': len(target) if target else 0,
               'tgt_before': float(sum(target.values())) if target else 0.0,
               'scale': 1.0, 'tgt_total': float(sum(target.values())) if target else 0.0,
               'invisible_codes': ''}
        if os.environ.get('QUANT_DBG_SCALE') == '1':
            _vis = set(cur_pos.keys())
            _inv = [stock_codes[j] for j in range(len(stock_codes))
                    if positions[j] > 0 and not _vis.__contains__(stock_codes[j])]
            _dg['invisible_codes'] = '|'.join(_inv[:12])
            _dg['n_invisible'] = len(_inv)

        # 现金流约束: target按total_equity算但实际只有cash可用, 缩放total到cash能支持的范围
        if target:
            _target_total = sum(target.values())
            _cur_pos_value = float(np.dot(positions.astype(np.float64), np.nan_to_num(px_today, 0)))
            _available = cash + _cur_pos_value  # max deployable = cash + current holdings
            if _target_total > _available * 0.98:
                _scale = (_available * 0.98) / max(_target_total, 1.0)
                target = {c: v * _scale for c, v in target.items()}
                _dg['scale'] = float(_scale)
                _dg['tgt_total'] = float(_target_total)
                _dg['available'] = float(_available)

        # 记录选股
        for c, tv in target.items():
            if c in prices and c not in cur_pos:
                sig_score = 0.0
                sig = strategy.signal_store.get(c, date)
                if sig is not None:
                    sig_score = float(getattr(sig, 'score', 0.0))
                selections.append({'date': date, 'code': c,
                                   'weight': tv / max(nav[i], 1.0),
                                   'score': sig_score})

        # 诊断落盘(env门控, 只写/tmp)
        if os.environ.get('QUANT_DBG_SCALE') == '1':
            try:
                import csv as _csvd
                _dg['sum_sel_w'] = float(sum(s['weight'] for s in
                                             strategy.portfolio.last_selection))
                _dg.setdefault('available', float(nav[i]))
                _p = '/tmp/scale_diag.csv'
                _new = not os.path.exists(_p)
                with open(_p, 'a', newline='') as _fd:
                    _wd = _csvd.DictWriter(_fd, fieldnames=sorted(_dg.keys()))
                    if _new:
                        _wd.writeheader()
                    _wd.writerow(_dg)
            except Exception as _e:
                print(f' [scale_diag] {_e}')

        # 调仓执行 (B批修复B3): 目标只入挂单簿, 次日开盘成交; 当日收盘只决策不成交
        _pending = target

        # 跟踪回撤
        running_peak = np.maximum.accumulate(nav[:i+1])
        running_dd = (nav[i] - running_peak[-1]) / max(running_peak[-1], 1.0)
        # 记录每日状态 (每20个交易日采样一次)
        if i % 20 == 0 or i == n_dates - 1:
            pos_count = int(np.sum(positions > 0))
            exposure_val = float(np.dot(positions.astype(np.float64), np.nan_to_num(px_today, 0))) / max(nav[i], 1.0)
            plog.log_backtest_daily(calendar[i].date(), nav[i], float(running_dd),
                                    pos_count, exposure_val, float(daily_ret[i-1]) if i > 0 else 0.0)

    # 记录执行成交汇总
    plog.log_execution(
        buy_filled=_fill_stats['buy_filled'],
        sell_filled=_fill_stats['sell_filled'],
        buy_attempted=_fill_stats['buy_attempted'],
        sell_attempted=_fill_stats['sell_attempted'],
        buy_limit_skip=_fill_stats['buy_limit_up_skip'],
        sell_limit_skip=_fill_stats['sell_limit_down_skip'],
        cash_insufficient=_fill_stats['buy_cash_insufficient'],
        tplus1_blocked=_fill_stats['sell_tplus1_blocked'],
    )

    # 4. 计算指标
    final_nav = nav[-1]
    total_return = final_nav / initial_cash - 1.0
    daily_ret_clean = np.nan_to_num(daily_ret, 0.0)
    mean_daily = np.mean(daily_ret_clean)
    std_daily = np.std(daily_ret_clean)
    sharpe = mean_daily / max(std_daily, 1e-10) * np.sqrt(252)

    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    max_dd = float(np.min(dd))

    annual_rets = {}
    annual_mdd = {}
    for yr in sorted(set(d.year for d in calendar)):
        mask = np.array([d.year == yr for d in calendar])
        yr_nav = nav[mask]
        if len(yr_nav) > 1:
            annual_rets[yr] = float(yr_nav[-1] / yr_nav[0] - 1.0)
            yr_peak = np.maximum.accumulate(yr_nav)
            yr_dd = (yr_nav - yr_peak) / yr_peak
            annual_mdd[yr] = float(np.min(yr_dd))

    # 成交统计: 每个调仓日的买卖操作 ≈ 选股数 × 2
    n_trades = len(selections) * 2

    print(f"\n=== 向量化回测结果 ===")
    print(f"初始资金: {initial_cash:,.0f}")
    print(f"最终净值: {final_nav:,.0f}  (总收益 {total_return*100:.2f}%)")
    for yr, r in sorted(annual_rets.items()):
        mdd = annual_mdd.get(yr, 0)
        print(f"  {yr}: {r*100:.2f}%  (最大回撤 {abs(mdd)*100:.1f}%)")
    print(f"Sharpe: {sharpe:.4f}")
    print(f"最大回撤: {abs(max_dd)*100:.2f}%")
    print(f"选股记录: {len(selections)} 次")
    print(f"日收益序列: {len(daily_ret_clean)} 点, 均值={mean_daily:.6f}, std={std_daily:.6f}")
    # 成交率统计
    ba = max(_fill_stats['buy_attempted'], 1)
    sa = max(_fill_stats['sell_attempted'], 1)
    print(f"成交率: 买入{_fill_stats['buy_filled']}/{_fill_stats['buy_attempted']}={_fill_stats['buy_filled']/ba*100:.1f}%"
          f" 卖出{_fill_stats['sell_filled']}/{_fill_stats['sell_attempted']}={_fill_stats['sell_filled']/sa*100:.1f}%"
          f" 资金不足={_fill_stats['buy_cash_insufficient']}"
          f" 跳空罚={_fill_stats['gap_buy_penalized']}买/{_fill_stats['gap_sell_penalized']}卖")
    # 记录到诊断模块
    try:
        from analysis.backtest_diagnostics import get_diagnostics
        get_diagnostics().record_fill_stats(_fill_stats)
    except Exception as e:
        import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()

    # V-验证体系: 落盘净值曲线(date/nav/daily_ret), 供sharpe_validation.py做
    # 年化分解/滚动夏普/MC置信区间/回撤画像, 无需重跑即可分析
    # 注意: 不得在此处import pandas (会遮蔽模块级pd, 引发UnboundLocalError)
    try:
        _eq_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'rolling_validation_results', 'equity_curve.csv')
        pd.DataFrame({
            'date': [pd.Timestamp(d).strftime('%Y-%m-%d') for d in calendar],
            'nav': nav,
            'daily_ret': np.concatenate([[np.nan], daily_ret]),
        }).to_csv(_eq_path, index=False)
        print(f"净值曲线已保存: {_eq_path}")
        # H4证据(2026-09-06): 逐日状态落盘(regime/n_pos/exposure),
        # 供分析FAST预警期的暴露行为与回撤关系(positions_dist每日1次调用, 顺序=calendar)
        _rs = getattr(getattr(strategy, 'portfolio', None), '_dbg', {}).get('positions_dist', [])
        if _rs:
            _rs = _rs[:len(calendar)]
            _rs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'rolling_validation_results', 'regime_state.csv')
            pd.DataFrame({
                'date': [pd.Timestamp(d).strftime('%Y-%m-%d') for d in calendar[:len(_rs)]],
                'regime': [r[0] for r in _rs],
                'n_pos': [r[1] for r in _rs],
                'exposure': [r[2] for r in _rs],
            }).to_csv(_rs_path, index=False)
            print(f"逐日状态已保存: {_rs_path}")
    except Exception as e:
        import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()

    return {'nav': nav, 'daily_returns': daily_ret_clean, 'sharpe': sharpe,
            'max_drawdown': max_dd, 'annual_returns': annual_rets,
            'annual_max_drawdown': annual_mdd,
            'final_value': final_nav, 'selections': selections,
            'n_dates': n_dates, 'realized_trades': _realized_trades,
            'fill_stats': _fill_stats}


if __name__ == "__main__":
    # === 日志输出：同时输出到 stdout 和文件，方便监控进度 ===
    import sys as _sys
    from datetime import datetime as _dt
    _log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f'bt_execution_{_dt.now().strftime("%Y%m%d_%H%M%S")}.log')
    class _Tee:
        def __init__(self, *files): self.files = files
        def write(self, data):
            for f in self.files: f.write(data); f.flush()
        def flush(self):
            for f in self.files: f.flush()
    _log_f = open(_log_path, 'w', encoding='utf-8')
    _sys.stdout = _Tee(_sys.stdout, _log_f)
    print(f"回测日志: {_log_path}")
    # === 环境指纹 (2026-09-12): 三指纹只覆盖文件, 解释器/库版本曾是A/B分歧根因 ===
    try:
        import platform as _platform
        import numpy as _np, pandas as _pd, xgboost as _xgb
        print(f"环境指纹: python={_sys.executable} "
              f"numpy={_np.__version__} pandas={_pd.__version__} "
              f"xgboost={_xgb.__version__} platform={_platform.platform()[:60]}")
    except Exception as _e:
        print(f"环境指纹: 获取失败 {_e}")

    # 加载基本面数据 (支持 _qfq.csv 和 _hfq.csv)
    stock_pool_enabled = config.get('stock_pool.enabled', True)
    stock_codes = []
    for f in os.listdir(DATA_PATH):
        if f.startswith('._'):
            continue
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            stock_codes.append(f.replace('_qfq.csv', ''))
        elif f.endswith('_hfq.csv') and f != 'sh000001_hfq.csv':
            stock_codes.append(f.replace('_hfq.csv', ''))

    # 股票池过滤
    membership_map = _load_pool_membership()
    if stock_pool_enabled:
        if membership_map is not None:
            _union = set()
            for _codes in membership_map.values():
                _union |= _codes
            stock_codes = [c for c in stock_codes if c in _union]
            print(f"基本面数据加载(日历union池): {len(stock_codes)} 只")
        else:
            stock_pool = get_stock_pool(todate=_pool_todate(),
                                        bse_exclude=config.get('stock_pool.bse_exclude', True))
            stock_codes = [c for c in stock_codes if c in stock_pool]
            print(f"基本面数据加载(股票池): {len(stock_codes)} 只")
    else:
        print(f"基本面数据加载(全市场): {len(stock_codes)} 只")

    # 科创板(688xxx) 不再排除 — 与 add_data_and_signal 策略一致
    fundamental_data = FundamentalData(FUNDAMENTAL_PATH, stock_codes)

    # 初始化情绪分析编排器
    sentiment_orch = None
    sentiment_enabled = config.get('industry_sentiment.enabled', False)
    if sentiment_enabled and SENTIMENT_AVAILABLE:
        print("[Sentiment] 初始化情绪分析模块...")
        try:
            sentiment_orch = SentimentOrchestrator(config, backtest_mode=True)
        except Exception as e:
            print(f"[Sentiment] 初始化异常 (将跳过情绪调整): {e}")
            sentiment_orch = None

    cerebro = bt.Cerebro()
    strategy = Strategy(
        init_cash=CASH,
        fundamental_data=fundamental_data,
        sentiment_orchestrator=sentiment_orch,
    )

    # 复用已有信号CSV(仅portfolio/执行变更时), 否则重新生成
    _signals_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'rolling_validation_results', 'backtest_signals.csv')
    if os.path.exists(_signals_csv) and not _signals_stale(_signals_csv):
        print(f"复用已有信号: {_signals_csv}")
        strategy.signal_store.finalize(_signals_csv, membership_map=membership_map)
        _idx_path = os.path.join(DATA_PATH, 'sh000001_qfq.csv')
        if not os.path.exists(_idx_path):
            _idx_path = os.path.join(DATA_PATH, '000001_qfq.csv')
        if os.path.exists(_idx_path):
            import pandas as pd
            _idx_df = pd.read_csv(_idx_path, parse_dates=['datetime'])
            if FROMDATE: _idx_df = _idx_df[_idx_df['datetime'] >= FROMDATE]
            if TODATE: _idx_df = _idx_df[_idx_df['datetime'] <= TODATE]
            strategy.generate_market_regime(_idx_df)
            del _idx_df
    else:
        # 不再预删既有CSV: add_data_and_signal原子写(os.replace)自然覆盖,
        # 中途被杀时旧CSV保留可复用 (2026-09-06)
        add_data_and_signal(cerebro, strategy, fundamental_data,
                            membership_map=membership_map)
        # 记录生成此CSV的信号代码指纹 → 下次复用门禁比对
        _sidecar = os.path.join(os.path.dirname(os.path.abspath(_signals_csv)),
                                '.signal_code_fp')
        try:
            with open(_sidecar, 'w') as _f:
                # E-B1v2: sidecar协议=fp|warmup_days (warmup影响信号内容, 指纹不可见)
                _f.write(f"{_signal_code_fingerprint()}|{_B1V2_WARMUP_DAYS}")
        except OSError as _e:
            print(f"[WARN] 信号代码指纹sidecar写入失败: {_e}")

    # ======  向量化回测引擎 (替代 backtrader 逐 bar 循环) ======
    del cerebro
    gc.collect()
    _malloc_trim()
    result = _vectorized_backtest(strategy, fundamental_data, FROMDATE, TODATE, CASH)

    # DEBUG: 打印portfolio行为统计
    if hasattr(strategy, 'portfolio') and hasattr(strategy.portfolio, 'print_dbg_summary'):
        strategy.portfolio.print_dbg_summary()

    # 保存选股结果
    if result['selections']:
        selections_df = pd.DataFrame(result['selections'])
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        selections_path = os.path.join(strategy_dir, 'rolling_validation_results', 'portfolio_selections.csv')
        os.makedirs(os.path.dirname(selections_path), exist_ok=True)
        selections_df.to_csv(selections_path, index=False)
        print(f"\n选股结果已保存: {len(selections_df)} 条 -> {selections_path}")

    # 保存逐笔平仓审计
    if result.get('realized_trades'):
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        trades_path = os.path.join(strategy_dir, 'rolling_validation_results', 'trade_realized.csv')
        pd.DataFrame(result['realized_trades']).to_csv(trades_path, index=False)
        print(f"逐笔平仓审计已保存: {len(result['realized_trades'])} 笔 -> {trades_path}")
    _fs = result.get('fill_stats', {})
    print(f"成交统计: 买{ _fs.get('buy_filled', 0)}/{_fs.get('buy_attempted', 0)} "
          f"(涨停跳过{_fs.get('buy_limit_up_skip', 0)}, 现金不足{_fs.get('buy_cash_insufficient', 0)}, "
          f"限价成交{_fs.get('tranche_filled', 0)}, 限价现金不足{_fs.get('tranche_cash_insufficient', 0)}) "
          f"卖{_fs.get('sell_filled', 0)}/{_fs.get('sell_attempted', 0)} "
          f"(跌停跳过{_fs.get('sell_limit_down_skip', 0)}, T+1拦截{_fs.get('sell_tplus1_blocked', 0)})")

    # === 全链路追踪报告 ===
    plog.report()

    # === 妖股观察名单: 从信号中筛选潜在妖股候选 ===
    try:
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        signals_output_path = os.path.join(strategy_dir, 'rolling_validation_results', 'backtest_signals.csv')
        signals_df = pd.read_csv(signals_output_path, low_memory=False)
        yg = signals_df.copy()
        yg['volume_ratio'] = pd.to_numeric(yg['volume_ratio'], errors='coerce')
        yg['daily_return'] = pd.to_numeric(yg['daily_return'], errors='coerce')
        yg['score'] = pd.to_numeric(yg['score'], errors='coerce')
        # buy列修复: CSV中为字符串 'True'/'False', 需转bool
        yg['buy_bool'] = yg['buy'].astype(str).str.lower().map({'true': True, 'false': False})

        # 妖股特征: 高量比(>0.3, arctan压缩后≈2x原始量比) + 高日收益(>3%) + 正score, 但未触发买入
        yg_candidates = yg[
            (yg['volume_ratio'] > 0.3) &
            (yg['daily_return'] > 0.03) &
            (yg['score'] > 0) &
            (yg['buy_bool'] == False)
        ].copy()

        if len(yg_candidates) > 0:
            yg_candidates = yg_candidates.sort_values(['date', 'score'], ascending=[True, False])
            yg_path = os.path.join(strategy_dir, 'rolling_validation_results', 'yaogu_watchlist.csv')
            yg_candidates[['date', 'code', 'score', 'volume_ratio', 'daily_return',
                            'factor_name', 'industry']].to_csv(yg_path, index=False)
            print(f"妖股观察名单已保存: {len(yg_candidates)} 条 -> {yg_path}")
        del yg, signals_df
    except Exception as e:
        print(f"[妖股名单] 生成失败: {e}")
        import traceback; traceback.print_exc()
