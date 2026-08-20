#!/usr/bin/env python3
"""
实盘选股 Dashboard — Flask 后端.

用法:
    .venv/bin/python strategy/dashboard_server.py [--port 5000]
    # 浏览器打开 http://localhost:5000
"""

import os, sys, json, argparse, io, subprocess, threading
from datetime import date as date_type, datetime

import pandas as pd
import numpy as np
from flask import Flask, jsonify, send_file, request

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_DIR, 'data', 'stock_data', 'backtrader_data')
TEMPLATE_DIR = os.path.join(_SCRIPT_DIR, 'templates')
BAR_COUNT = 120

app = Flask(__name__, template_folder=TEMPLATE_DIR)


# ── helpers ──────────────────────────────────────────────────────────

def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


_stock_names = None

def _get_stock_names():
    """懒加载股票代码→名称映射 (去除 XD/XR/DR 除权除息前缀)."""
    global _stock_names
    if _stock_names is None:
        fp = os.path.join(_PROJECT_DIR, 'data', 'stock_data', 'stock_metadata', 'stock_list.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp, dtype={'symbol': str}, usecols=['symbol', 'name'])
            _stock_names = {
                code: str(n).lstrip('XDXRDR') if str(n)[:2] in ('XD', 'XR', 'DR') else str(n)
                for code, n in zip(df['symbol'], df['name'])
            }
        else:
            _stock_names = {}
    return _stock_names


def _compute_channel(closes, highs, lows, total_bars):
    """用线性回归计算趋势通道 (上升/下降通道).

    用最近40根K线的高低点中点做回归, 取近端偏离度定通道宽度.
    前面补 None 使数组长度匹配 total_bars.
    """
    n = min(len(closes), 40)
    c = closes[-n:]
    h = highs[-n:]
    l = lows[-n:]
    x = np.arange(n)
    mid = (h + l) / 2
    slope, intercept = np.polyfit(x, mid, 1)
    center = slope * x + intercept
    recent = min(20, n)
    upper_dev = float(np.max(h[-recent:] - center[-recent:]))
    lower_dev = float(np.max(center[-recent:] - l[-recent:]))
    upper = center + upper_dev
    lower = center - lower_dev
    pad = total_bars - n
    nulls = [None] * pad
    return {
        'upper': nulls + [round(v, 2) for v in upper.tolist()],
        'center': nulls + [round(v, 2) for v in center.tolist()],
        'lower': nulls + [round(v, 2) for v in lower.tolist()],
        'slope': round(float(slope), 4),
        'upper_width': round(upper_dev, 2),
        'lower_width': round(lower_dev, 2),
        'type': 'ascending' if slope > 0 else 'descending' if slope < 0 else 'flat',
    }


def _get_latest_close(code):
    """读取某只股票最新 close 价格."""
    fp = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(fp):
        return None
    try:
        row = pd.read_csv(fp, usecols=['close']).iloc[-1]
        return float(row['close'])
    except Exception:
        return None


def _get_index_regime():
    """获取当前市场状态."""
    sys.path.insert(0, _SCRIPT_DIR)
    from core.market_regime_detector import MarketRegimeDetector
    idx_path = os.path.join(DATA_DIR, 'sh000001_qfq.csv')
    if not os.path.exists(idx_path):
        return {'regime': 0, 'name': '未知', 'momentum_score': 0, 'trend_score': 0}
    df = pd.read_csv(idx_path, parse_dates=['datetime'])
    detector = MarketRegimeDetector()
    result = detector.generate(df)
    last = result.iloc[-1]
    names = {1: '牛市', 0: '中性', -1: '熊市'}
    return {
        'regime': int(last.get('regime', 0)),
        'name': names.get(int(last.get('regime', 0)), '未知'),
        'momentum_score': float(last.get('momentum_score', 0)),
        'trend_score': float(last.get('trend_score', 0)),
    }


# ── API routes ───────────────────────────────────────────────────────

@app.route('/')
def index():
    with open(os.path.join(TEMPLATE_DIR, 'dashboard.html'), 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/api/portfolio')
def api_portfolio():
    """当前持仓 + 盈亏."""
    positions_path = os.path.join(_PROJECT_DIR, 'current_positions.json')
    orders_path = os.path.join(_PROJECT_DIR, 'trade_orders.json')

    if not os.path.exists(positions_path):
        return jsonify({'positions': [], 'total_value': 0, 'total_cost': 0, 'pnl': 0, 'pnl_pct': 0})

    positions = _load_json(positions_path)
    orders_data = _load_json(orders_path) if os.path.exists(orders_path) else {'orders': []}

    # 构建 code → order 映射
    order_map = {}
    for o in orders_data.get('orders', []):
        if o.get('action') in ('open', 'adjust'):
            code = o['stock_code'].split('.')[0]
            order_map[code] = o

    # 从信号文件获取行业映射 (code → industry)
    ind_map = {}
    try:
        signals_dir = os.path.join(_SCRIPT_DIR, 'rolling_validation_results')
        sig_files = [f for f in os.listdir(signals_dir)
                     if f.startswith('backtest_signals') and f.endswith('.csv')]
        if sig_files:
            sf = pd.read_csv(os.path.join(signals_dir, sig_files[0]),
                            dtype={'code': str}, usecols=['code', 'industry', 'date'])
            sf = sf.sort_values('date').groupby('code').last()
            ind_map = sf['industry'].to_dict()
    except Exception:
        pass

    result = []
    total_value = 0
    total_cost = 0
    today_str = date_type.today().isoformat()
    names = _get_stock_names()

    for code, pos in positions.items():
        entry_price = pos['entry_price']
        shares = pos['shares']
        amount = pos['amount']
        latest_close = _get_latest_close(code)

        if latest_close is None:
            latest_close = entry_price

        market_value = shares * latest_close
        pnl = market_value - amount
        pnl_pct = (latest_close - entry_price) / entry_price * 100

        entry_date = pos.get('entry_date', '')
        days_held = ''
        if entry_date:
            try:
                days_held = (date_type.today() - date_type.fromisoformat(entry_date)).days
            except Exception:
                pass

        order = order_map.get(code, {})
        item = {
            'code': code,
            'name': names.get(code, ''),
            'entry_price': entry_price,
            'latest_price': latest_close,
            'shares': shares,
            'amount': amount,
            'market_value': market_value,
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'stop_loss': order.get('stop_loss_price'),
            'take_profit': order.get('take_profit_price'),
            'entry_date': entry_date,
            'days_held': days_held,
            'industry': ind_map.get(code, ''),
        }
        result.append(item)
        total_value += market_value
        total_cost += amount

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    return jsonify({
        'positions': result,
        'total_value': round(total_value, 2),
        'total_cost': round(total_cost, 2),
        'pnl': round(total_pnl, 2),
        'pnl_pct': round(total_pnl_pct, 2),
        'count': len(result),
    })


@app.route('/api/orders')
def api_orders():
    """最新订单."""
    orders_path = os.path.join(_PROJECT_DIR, 'trade_orders.json')
    if not os.path.exists(orders_path):
        return jsonify({'date': '', 'orders': []})
    data = _load_json(orders_path)
    return jsonify(data)


@app.route('/api/kline/<code>')
def api_kline(code):
    """某只股票的 K 线数据 (最近 BAR_COUNT 天), ECharts candlestick 格式."""
    fp = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(fp):
        return jsonify({'error': f'{code} 数据不存在'}), 404

    df = pd.read_csv(fp, parse_dates=['datetime'])
    df = df.tail(BAR_COUNT)

    # ECharts candlestick: [open, close, low, high]
    ohlc = []
    volumes = []
    dates = []
    for i, (_, row) in enumerate(df.iterrows()):
        dates.append(row['datetime'].strftime('%Y-%m-%d'))
        ohlc.append([
            float(row['open']),
            float(row['close']),
            float(row['low']),
            float(row['high']),
        ])
        volumes.append([
            i,
            float(row['volume']),
            1 if row['close'] >= row['open'] else -1,
        ])

    # 计算趋势通道 (线性回归)
    closes = df['close'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    channel = _compute_channel(closes, highs, lows, len(dates))

    return jsonify({
        'code': code,
        'dates': dates,
        'ohlc': ohlc,
        'volumes': volumes,
        'channel': channel,
    })


@app.route('/api/market')
def api_market():
    """市场状态."""
    return jsonify(_get_index_regime())


# ── 后台任务管理 ─────────────────────────────────────────────────────

_task_state = {
    'running': False,
    'step': '',
    'process': None,
    'log_file': '',
    'started_at': '',
    'error': '',
}

_VENV_PYTHON = os.path.join(_PROJECT_DIR, '.venv', 'bin', 'python')
_LOG_DIR = os.path.join(_SCRIPT_DIR, 'logs')


def _run_in_background(step):
    """在后台线程中执行 run_live.py."""
    global _task_state
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(_LOG_DIR, f'live_{ts}.log')
    os.makedirs(_LOG_DIR, exist_ok=True)

    # 构建命令
    if step == 'data':
        cmd = [_VENV_PYTHON, os.path.join(_SCRIPT_DIR, 'run_live.py'),
               '--skip-signals']
    elif step == 'signals':
        cmd = [_VENV_PYTHON, os.path.join(_SCRIPT_DIR, 'run_live.py'),
               '--skip-data']
    else:  # 'live' — 一键选股
        cmd = [_VENV_PYTHON, os.path.join(_SCRIPT_DIR, 'run_live.py')]

    _task_state['running'] = True
    _task_state['step'] = step
    _task_state['log_file'] = log_file
    _task_state['started_at'] = datetime.now().strftime('%H:%M:%S')
    _task_state['error'] = ''

    try:
        with open(log_file, 'w') as f:
            f.write(f'[{datetime.now().strftime("%H:%M:%S")}] 开始执行: {" ".join(cmd)}\n')
            f.flush()
            proc = subprocess.Popen(cmd, cwd=_PROJECT_DIR, stdout=f, stderr=subprocess.STDOUT)
            _task_state['process'] = proc
            proc.wait()
            status = '成功' if proc.returncode == 0 else f'失败 (exit={proc.returncode})'
            f.write(f'\n[{datetime.now().strftime("%H:%M:%S")}] 任务{status}\n')
    except Exception as e:
        _task_state['error'] = str(e)
        try:
            with open(log_file, 'a') as f:
                f.write(f'\n[ERROR] {e}\n')
        except Exception:
            pass
    finally:
        _task_state['running'] = False
        _task_state['process'] = None


@app.route('/api/run/<step>', methods=['POST'])
def api_run(step):
    """启动后台任务: data / signals / live."""
    if step not in ('data', 'signals', 'live'):
        return jsonify({'error': f'无效步骤: {step}'}), 400

    if _task_state['running']:
        return jsonify({'error': '已有任务正在运行', 'running': True,
                        'step': _task_state['step'], 'started_at': _task_state['started_at']}), 409

    t = threading.Thread(target=_run_in_background, args=(step,), daemon=True)
    t.start()

    return jsonify({'status': 'started', 'step': step,
                    'started_at': _task_state['started_at']})


@app.route('/api/run/stop', methods=['POST'])
def api_run_stop():
    """停止正在运行的后台任务."""
    if not _task_state['running']:
        return jsonify({'error': '没有正在运行的任务'}), 400
    proc = _task_state.get('process')
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _task_state['running'] = False
    _task_state['process'] = None
    with open(_task_state['log_file'], 'a') as f:
        f.write(f'\n[{datetime.now().strftime("%H:%M:%S")}] 用户手动停止\n')
    return jsonify({'status': 'stopped'})


@app.route('/api/run/status')
def api_run_status():
    """查询当前任务状态 + 最近日志."""
    log_lines = []
    if _task_state['log_file'] and os.path.exists(_task_state['log_file']):
        try:
            with open(_task_state['log_file'], 'r') as f:
                lines = f.readlines()
                # 返回最后 80 行, 只保留有意义的行
                log_lines = [l.rstrip() for l in lines[-80:]
                           if l.strip() and 'loading:' not in l
                           and '\rit/s' not in l]
        except Exception:
            pass

    return jsonify({
        'running': _task_state['running'],
        'step': _task_state['step'],
        'started_at': _task_state['started_at'],
        'error': _task_state['error'],
        'log': log_lines,
    })


# ── 新增 API ─────────────────────────────────────────────────────────

def _get_index_data(code, name):
    """通用指数数据获取 — 兼容 sh000001(英文列名) 和 399006/000688(中文列名)."""
    raw_dir = os.path.join(_PROJECT_DIR, 'data', 'stock_data', 'raw_data')
    fp = os.path.join(raw_dir, code, 'qfq.csv')
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp).tail(60)
    # 统一列名
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ('date', '日期'): col_map[c] = 'date'
        elif cl in ('open', '开盘'): col_map[c] = 'open'
        elif cl in ('close', '收盘'): col_map[c] = 'close'
        elif cl in ('high', '最高'): col_map[c] = 'high'
        elif cl in ('low', '最低'): col_map[c] = 'low'
    df = df.rename(columns=col_map)
    df['date'] = pd.to_datetime(df['date'])
    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
    ohlc = [[float(r['open']), float(r['close']), float(r['low']), float(r['high'])]
            for _, r in df.iterrows()]
    last = df.iloc[-1]
    prev = df.iloc[-2]
    chg = float(last['close'] - prev['close'])
    chg_pct = round(float(chg / prev['close'] * 100), 2)
    amp = round(float((last['high'] - last['low']) / prev['close'] * 100), 2)
    return {
        'code': code, 'name': name,
        'dates': dates, 'ohlc': ohlc,
        'close': float(last['close']), 'change': round(chg, 2),
        'change_pct': chg_pct, 'amplitude': amp,
    }

_index_map = [
    ('sh000001', '上证指数'),
    ('399006', '创业板指'),
]

@app.route('/api/index')
def api_index():
    """全部指数数据."""
    results = []
    for code, name in _index_map:
        d = _get_index_data(code, name)
        if d: results.append(d)
    return jsonify(results)


@app.route('/api/sectors')
def api_sectors():
    """概念板块: Top 10 涨 + Top 10 跌."""
    fp = os.path.join(_PROJECT_DIR, 'data', 'stock_data', 'concept_daily.csv')
    if not os.path.exists(fp):
        return jsonify({'top_gainers': [], 'top_losers': []})
    df = pd.read_csv(fp)
    df = df.sort_values('涨跌幅', ascending=False)
    top = df.head(10)
    bottom = df.tail(10)
    def row(r):
        return {'name': str(r['板块名称']), 'change_pct': float(r['涨跌幅']),
                'leader': str(r.get('领涨股票', '')), 'leader_pct': float(r.get('领涨股票-涨跌幅', 0))}
    return jsonify({
        'top_gainers': [row(r) for _, r in top.iterrows()],
        'top_losers': [row(r) for _, r in bottom.iterrows()],
    })


@app.route('/api/config')
def api_config():
    """关键风控/策略参数."""
    config_path = os.path.join(_SCRIPT_DIR, 'config', 'factor_config.yaml')
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    p = cfg.get('portfolio', {})
    s = p.get('selection', {})
    ps = p.get('params', {})
    return jsonify({
        'factor_mode': cfg.get('factor_mode', 'unknown'),
        'max_positions': ps.get('max_positions', '-'),
        'position_stop_loss': f"{p.get('position_stop_loss', 0)*100:.0f}%",
        'portfolio_stop_loss': f"{p.get('portfolio_stop_loss', 0)*100:.0f}%",
        'target_volatility': f"{p.get('target_volatility', 0)*100:.0f}%",
        'min_confidence': s.get('min_confidence', '-'),
        'min_hold_days': ps.get('min_hold_days', '-'),
        'base_exposure': p.get('base_exposure', '-'),
        'buy_threshold': cfg.get('signal', {}).get('buy_threshold', '-'),
        'rebalance_days': cfg.get('backtest', {}).get('rebalance_days', '-'),
        'cash': cfg.get('backtest', {}).get('cash', '-'),
    })


# ═══════════════════════════════════════════════════════════════════
# 文章工作台 (media workbench)
# ═══════════════════════════════════════════════════════════════════

MEDIA_OUT_DIR = os.path.join(_SCRIPT_DIR, 'media_out')
MEDIA_IMG_DIR = os.path.join(MEDIA_OUT_DIR, 'images')


@app.route('/media')
def media_page():
    """文章工作台页面."""
    from flask import render_template
    return render_template('media.html')


@app.route('/showcase')
def showcase_page():
    """展示页 — 用于截图到公众号文章."""
    from flask import render_template
    return render_template('showcase.html')


def _selection_close(code, entry_date):
    """选股日收盘价 = 建仓日前最后一个交易日收盘价."""
    fp = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_csv(fp, parse_dates=['datetime'])
        df = df[df['datetime'] < pd.Timestamp(entry_date)]
        if df.empty:
            return None
        return float(df['close'].iloc[-1])
    except Exception:
        return None


def _sel_date_of(code, entry_date):
    """建仓日对应的选股日 (前一交易日)."""
    fp = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_csv(fp, parse_dates=['datetime'])
        df = df[df['datetime'] < pd.Timestamp(entry_date)]
        if df.empty:
            return None
        return df['datetime'].iloc[-1].strftime('%m-%d')
    except Exception:
        return None


@app.route('/api/media/overview')
def api_media_overview():
    """工作台总览: 市场 + 今日选股 + 持仓 + 批次统计."""
    # 市场概况
    idx_fp = os.path.join(DATA_DIR, 'sh000001_qfq.csv')
    market = None
    if os.path.exists(idx_fp):
        df = pd.read_csv(idx_fp, parse_dates=['datetime'])
        if len(df) >= 6:
            t = df.iloc[-1]
            p = df.iloc[-2]
            market = {
                'date': t['datetime'].strftime('%Y-%m-%d'),
                'close': round(float(t['close']), 2),
                'chg': round((float(t['close']) - float(p['close'])) / float(p['close']) * 100, 2),
                'chg_5d': round((float(t['close']) - float(df['close'].iloc[-6])) / float(df['close'].iloc[-6]) * 100, 2),
            }

    # 今日选股 (最新 trade_orders)
    orders_fp = os.path.join(_PROJECT_DIR, 'trade_orders.json')
    orders_date, opens, adjusts = None, [], []
    if os.path.exists(orders_fp):
        try:
            od = _load_json(orders_fp)
            orders_date = od.get('date')
            names = _get_stock_names()
            for o in od.get('orders', []):
                code = o['stock_code'].split('.')[0]
                entry = None
                if o.get('action') == 'open':
                    # 成本基准: 选股日(订单日前一交易日)收盘价
                    entry = _selection_close(code, od.get('date'))
                row = {
                    'code': code,
                    'name': names.get(code, ''),
                    'action': o.get('action'),
                    'amount': o.get('amount'),
                    'stop_loss': o.get('stop_loss_price'),
                    'take_profit': o.get('take_profit_price'),
                    'entry': round(entry, 2) if entry else None,
                }
                (opens if o.get('action') == 'open' else adjusts).append(row)
        except Exception:
            pass

    # 持仓 + 批次统计 (成本基准: 选股日收盘价, 跟踪窗口: 5个交易日)
    HOLD_DAYS = 5  # 调仓周期: 每批选股跟踪5个交易日
    positions_fp = os.path.join(_PROJECT_DIR, 'current_positions.json')
    positions, batches = [], {}
    if os.path.exists(positions_fp):
        try:
            pos = _load_json(positions_fp)
            names = _get_stock_names()
            for code, p in sorted(pos.items(), key=lambda x: x[1].get('entry_date', '')):
                entry_date = p.get('entry_date')
                fp = os.path.join(DATA_DIR, f'{code}_qfq.csv')
                if not os.path.exists(fp):
                    continue
                d = pd.read_csv(fp, parse_dates=['datetime'])
                if d.empty:
                    continue
                sel_date = _sel_date_of(code, entry_date)
                sel_mask = d['datetime'] < pd.Timestamp(entry_date)
                if not sel_mask.any():
                    continue
                cost = float(d.loc[sel_mask, 'close'].iloc[-1])
                sel_idx = int(sel_mask.sum()) - 1  # 选股日在数据中的位置
                # 5日窗口: 选股日收盘 → 第5个交易日收盘 (数据不足时用最新)
                day5_idx = min(sel_idx + HOLD_DAYS, len(d) - 1)
                ref = float(d['close'].iloc[day5_idx])
                ref_date = d['datetime'].iloc[day5_idx].strftime('%Y-%m-%d')
                days_elapsed = day5_idx - sel_idx
                # 已完成: 用5日收盘; 进行中: 也用5日收盘(若已过), 否则最新
                pnl = (ref - cost) / cost * 100
                last = float(d['close'].iloc[-1])
                positions.append({
                    'code': code,
                    'name': names.get(code, ''),
                    'sel_date': sel_date or entry_date,
                    'entry_date': entry_date,
                    'amount': p.get('amount'),
                    'cost': round(cost, 2),
                    'last': round(last, 2),
                    'pnl': round(pnl, 2),
                    'pnl_str': f'{pnl:+.1f}%',
                    'days_elapsed': days_elapsed,
                    'done': days_elapsed >= HOLD_DAYS,
                    'ref_date': ref_date,
                })
                if sel_date:
                    batches.setdefault(sel_date, []).append(pnl)

            batch_stats = []
            sorted_batches = sorted(batches.items())
            # 最近10个批次单独展示, 更早的合并为"更早批次累计"
            recent, older = sorted_batches[-10:], sorted_batches[:-10]
            for d, ps in recent:
                # 该批次的持有天数 (取第一批股票的天数状态)
                bpos = [p for p in positions if p['sel_date'] == d]
                days = max((p['days_elapsed'] for p in bpos), default=0)
                batch_stats.append({
                    'date': d,
                    'n': len(ps),
                    'avg': round(sum(ps) / len(ps), 2),
                    'win': sum(1 for p in ps if p > 0),
                    'best': round(max(ps), 1),
                    'worst': round(min(ps), 1),
                    'days': days,
                    'done': days >= HOLD_DAYS,
                })
            if older:
                old_ps = [p for _, ps in older for p in ps]
                batch_stats.append({
                    'date': f'更早{len(older)}批',
                    'n': len(old_ps),
                    'avg': round(sum(old_ps) / len(old_ps), 2),
                    'win': sum(1 for p in old_ps if p > 0),
                    'best': round(max(old_ps), 1),
                    'worst': round(min(old_ps), 1),
                    'days': HOLD_DAYS,
                    'done': True,
                })
            # 整体统计
            all_pnl = [p for ps in batches.values() for p in ps]
            # 亏损票金额占比: 证明风险预算有效 (亏损的票通常配得少)
            loss_amt = sum(p['amount'] or 0 for p in positions if (p['pnl'] or 0) < 0)
            win_amt = sum(p['amount'] or 0 for p in positions if (p['pnl'] or 0) > 0)
            total_amt = loss_amt + win_amt
            total = {
                'n': len(all_pnl),
                'win': sum(1 for p in all_pnl if p > 0),
                'avg': round(sum(all_pnl) / len(all_pnl), 2) if all_pnl else 0,
                'loss_amt_pct': round(loss_amt / total_amt * 100, 1) if total_amt else 0,
                'total_amt': total_amt,
            }
        except Exception as e:
            total, batch_stats = {'n': 0, 'win': 0, 'avg': 0}, []

    return jsonify({
        'market': market,
        'orders_date': orders_date,
        'opens': opens,
        'adjusts': adjusts,
        'positions': positions,
        'batches': batch_stats,
        'total': total,
    })


@app.route('/api/media/article', methods=['POST'])
def api_media_generate_article():
    """生成每日复盘文章."""
    data = request.get_json(silent=True) or {}
    target_date = data.get('date')
    compliance = bool(data.get('compliance', False))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'media_article', os.path.join(_SCRIPT_DIR, 'media_article.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = mod.generate_review(target_date, compliance=compliance)
        if out and os.path.exists(out):
            with open(out, 'r', encoding='utf-8') as f:
                return jsonify({'ok': True, 'path': out, 'markdown': f.read()})
        return jsonify({'ok': False, 'error': '文章生成失败'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/media/charts', methods=['POST'])
def api_media_generate_charts():
    """生成文章配图."""
    try:
        script = os.path.join(_SCRIPT_DIR, 'article_charts.py')
        r = subprocess.run(
            [sys.executable, script], cwd=_SCRIPT_DIR,
            capture_output=True, text=True, timeout=120)
        imgs = sorted(os.listdir(MEDIA_IMG_DIR)) if os.path.exists(MEDIA_IMG_DIR) else []
        return jsonify({
            'ok': r.returncode == 0,
            'output': r.stdout[-500:] + r.stderr[-200:],
            'images': imgs,
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/media/article-latest')
def api_media_latest_article():
    """读取最近的复盘文章."""
    try:
        if not os.path.exists(MEDIA_OUT_DIR):
            return jsonify({'ok': False, 'error': 'media_out 目录不存在'})
        files = sorted(
            [f for f in os.listdir(MEDIA_OUT_DIR) if f.startswith('复盘_') and f.endswith('.md')])
        if not files:
            return jsonify({'ok': False, 'error': '还没有生成过复盘文章'})
        fp = os.path.join(MEDIA_OUT_DIR, files[-1])
        with open(fp, 'r', encoding='utf-8') as f:
            return jsonify({'ok': True, 'name': files[-1], 'markdown': f.read()})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/media/render', methods=['POST'])
def api_media_render():
    """把 Markdown 渲染成微信内联样式 HTML (本地排版, 替代外部 mdnice)."""
    data = request.get_json(silent=True) or {}
    markdown_text = data.get('markdown', '')
    theme = data.get('theme', 'minimal')
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'wechat_render', os.path.join(_SCRIPT_DIR, 'wechat_render.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if theme not in {th['id'] for th in mod.THEME_NAMES}:
            theme = 'minimal'
        html = mod.md_to_wechat(markdown_text, theme, img_root='/media-img')
        return jsonify({'ok': True, 'html': html, 'theme': theme})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/media-img/<path:subpath>')
def api_media_img(subpath):
    """文章配图静态路由: live_charts/日期/xxx.png 或 images/xxx.png."""
    try:
        if subpath.startswith('live_charts/'):
            root = os.path.join(_SCRIPT_DIR, 'live_charts')
            sub = subpath[len('live_charts/'):]
        else:
            root = MEDIA_IMG_DIR
            sub = os.path.basename(subpath)
        fp = os.path.realpath(os.path.join(root, sub))
        safe_root = os.path.realpath(root)
        if fp != safe_root and not fp.startswith(safe_root + os.sep):
            return jsonify({'ok': False, 'error': 'forbidden'}), 403
        if not os.path.exists(fp):
            return jsonify({'ok': False, 'error': 'not found'}), 404
        return send_file(fp)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/signals/<code>')
def api_signals(code):
    """查询某只股票在最新信号日期的信号详情."""
    signals_dir = os.path.join(_SCRIPT_DIR, 'rolling_validation_results')
    sig_files = [f for f in os.listdir(signals_dir) if f.startswith('backtest_signals') and f.endswith('.csv')]
    if not sig_files:
        sig_files = [f for f in os.listdir(signals_dir) if f == 'backtest_signals_exp0_new.csv']
    if not sig_files:
        return jsonify({'error': '信号文件不存在'})
    fp = os.path.join(signals_dir, sig_files[0])
    df = pd.read_csv(fp, dtype={'code': str})
    df = df[df['code'] == code]
    if df.empty:
        return jsonify({'error': f'{code} 无信号数据'})
    last = df.iloc[-1]
    return jsonify({
        'code': code,
        'date': str(last['date']),
        'buy': bool(last['buy']),
        'sell': bool(last['sell']),
        'score': round(float(last['score']), 4),
        'factor_name': str(last['factor_name']),
        'industry': str(last.get('industry', '')),
        'factor_quality': round(float(last.get('factor_quality', 0)), 4),
        'chan_buy_point': int(last.get('chan_buy_point', 0)),
        'signal_level': int(last.get('signal_level', 0)),
        'trend_type': int(last.get('trend_type', 0)),
        'mom_60d': round(float(last.get('mom_60d', 0)), 4),
        'dist_ma60': round(float(last.get('dist_ma60', 0)), 4),
        'risk_vol': round(float(last.get('risk_vol', 0)), 4),
        'daily_return': round(float(last.get('daily_return', 0)), 4),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='实盘选股 Dashboard')
    parser.add_argument('--port', type=int, default=5000, help='端口 (默认 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='绑定地址')
    args = parser.parse_args()

    print(f'实盘选股 Dashboard 启动: http://localhost:{args.port}')
    app.run(host=args.host, port=args.port, debug=False)
