"""
性能监控与结构化日志模块

提供:
  - @timeit 装饰器: 函数级耗时统计
  - memory(): 当前进程内存峰值(RSS)
  - ICHistory: 因子IC衰减趋势追踪
  - PipelineMonitor: 流水线级监控(信号生成/组合构建/因子选择)

用法:
  from core.monitor import monitor, timeit

  @timeit('signal.generate')
  def generate(...): ...

  monitor.memory('after factor calc')
  monitor.report()  # 输出汇总
"""

import json
import logging
import os
import sys
import time
import functools
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── 结构化日志配置 ──
_logger = logging.getLogger("quant")
_logger.setLevel(logging.DEBUG)

# 防止重复添加 handler（模块可能被多次导入）
if not _logger.handlers:
    # 控制台 handler: INFO+
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    _logger.addHandler(_ch)

    # 文件 handler: DEBUG+ (完整记录)
    _log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(_log_dir, exist_ok=True)
    _fh = logging.FileHandler(os.path.join(_log_dir, "quant.log"), encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _logger.addHandler(_fh)


def get_logger(name: str = None) -> logging.Logger:
    """获取结构化日志器"""
    if name:
        return _logger.getChild(name)
    return _logger


class ICHistory:
    """因子IC衰减趋势追踪

    按日期记录每个因子的截面IC, 提供滚动窗口统计和衰减检测。
    """

    def __init__(self, window_days: int = 60, decay_threshold: float = 0.02):
        self.window_days = window_days
        self.decay_threshold = decay_threshold
        # {factor_name: deque of (date, ic_value)}
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_days * 2))

    def record(self, factor_name: str, date, ic_value: float):
        """记录一次IC值"""
        self._history[factor_name].append((date, float(ic_value)))

    def get_stats(self, factor_name: str, recent_days: int = 20) -> dict:
        """获取因子近期IC统计"""
        records = list(self._history.get(factor_name, []))
        if not records:
            return {"n": 0, "ic_mean": 0, "ic_std": 0, "ir": 0, "decaying": False}

        recent = records[-recent_days:]
        ics = [r[1] for r in recent]
        ic_mean = float(np.mean(ics))
        ic_std = float(np.std(ics)) + 1e-10

        # 衰减检测: 后半段IC均值 < 前半段 - threshold
        mid = len(recent) // 2
        if mid >= 5:
            first_half = float(np.mean(ics[:mid]))
            second_half = float(np.mean(ics[mid:]))
            decaying = (second_half < first_half - self.decay_threshold)
        else:
            decaying = False

        return {
            "n": len(recent),
            "ic_mean": round(ic_mean, 5),
            "ic_std": round(ic_std, 5),
            "ir": round(ic_mean / ic_std, 4),
            "decaying": decaying,
            "trend": "↓" if decaying else ("↑" if second_half > first_half + self.decay_threshold else "→"),
        }

    def get_decaying_factors(self, min_n: int = 10) -> List[str]:
        """返回正在衰减的因子列表"""
        decaying = []
        for name in self._history:
            stats = self.get_stats(name)
            if stats["n"] >= min_n and stats["decaying"]:
                decaying.append(name)
        return decaying

    def summary(self) -> dict:
        """所有因子的IC汇总"""
        result = {}
        for name in sorted(self._history.keys()):
            result[name] = self.get_stats(name)
        return result


class PipelineMonitor:
    """流水线级性能监控

    追踪: 信号生成耗时、组合构建耗时、因子选择耗时、内存峰值、各阶段输出量。
    """

    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.memory_peaks: List[Tuple[str, float]] = []
        self.ic_history = ICHistory()
        self._start_time = time.time()

    def timeit(self, name: str):
        """装饰器: 记录函数耗时"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                t0 = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - t0
                self.timings[name].append(elapsed)
                _logger.debug(f"{name}: {elapsed:.2f}s")
                return result
            return wrapper
        return decorator

    def count(self, name: str, n: int = 1):
        """计数器"""
        self.counters[name] += n

    def memory(self, tag: str = ""):
        """记录当前内存峰值"""
        try:
            import resource
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            self.memory_peaks.append((tag, rss_mb))
            _logger.info(f"[MEM:{tag}] {rss_mb:.0f} MB")
        except Exception:
            pass

    def record_ic(self, factor_name: str, date, ic_value: float):
        """记录IC值到衰减追踪器"""
        self.ic_history.record(factor_name, date, ic_value)

    def get_timing_stats(self) -> dict:
        """耗时统计"""
        stats = {}
        for name, times in self.timings.items():
            arr = np.array(times)
            stats[name] = {
                "count": len(arr),
                "total_s": round(float(arr.sum()), 1),
                "mean_s": round(float(arr.mean()), 2),
                "max_s": round(float(arr.max()), 2),
                "p95_s": round(float(np.percentile(arr, 95)), 2),
            }
        return stats

    def report(self) -> dict:
        """生成监控报告"""
        elapsed = time.time() - self._start_time
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_elapsed_s": round(elapsed, 1),
            "timing": self.get_timing_stats(),
            "counters": dict(self.counters),
            "memory_peak_mb": round(max((m[1] for m in self.memory_peaks), default=0), 0),
            "memory_trace": [(tag, round(mb, 0)) for tag, mb in self.memory_peaks],
            "ic_decay": {
                "decaying_factors": self.ic_history.get_decaying_factors(),
                "summary": self.ic_history.summary(),
            },
        }
        return report

    def print_report(self):
        """打印监控报告到日志"""
        r = self.report()
        _logger.info("=" * 60)
        _logger.info("性能监控报告")
        _logger.info(f"  总耗时: {r['total_elapsed_s']:.1f}s")
        _logger.info(f"  内存峰值: {r['memory_peak_mb']:.0f} MB")
        _logger.info("  耗时分布:")
        for name, stats in r["timing"].items():
            _logger.info(f"    {name}: total={stats['total_s']}s, mean={stats['mean_s']}s, "
                        f"n={stats['count']}, p95={stats['p95_s']}s")
        _logger.info(f"  计数器: {r['counters']}")
        if r["ic_decay"]["decaying_factors"]:
            _logger.warning(f"  IC衰减因子: {r['ic_decay']['decaying_factors']}")
        _logger.info("=" * 60)
        return r

    def save_report(self, filepath: str = None):
        """持久化监控报告为 JSON（追加写入，跨交易日不丢失）

        Args:
            filepath: 目标文件路径，默认写入 logs/monitor_history.jsonl
        """
        if filepath is None:
            filepath = os.path.join(_log_dir, "monitor_history.jsonl")

        r = self.report()
        # 清理 IC summary 中的 numpy 类型以保证 JSON 可序列化
        clean_summary = {}
        for name, stats in r["ic_decay"]["summary"].items():
            clean_summary[name] = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in stats.items()
            }
        r["ic_decay"]["summary"] = clean_summary

        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
            _logger.info(f"监控报告已持久化: {filepath}")
        except Exception as e:
            _logger.warning(f"监控报告持久化失败: {e}")

    @staticmethod
    def load_history(filepath: str = None, n_recent: int = 20) -> list:
        """加载最近的监控历史记录

        Args:
            filepath: 目标文件路径
            n_recent: 返回最近 N 条记录

        Returns:
            list of dict, 最近的监控报告（按时间升序）
        """
        if filepath is None:
            filepath = os.path.join(_log_dir, "monitor_history.jsonl")

        records = []
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
        except Exception:
            pass
        return records[-n_recent:] if len(records) > n_recent else records


# ── 全局单例 ──
monitor = PipelineMonitor()


# ── 便捷函数 ──
def timeit(name: str):
    """独立装饰器, 委托给全局monitor"""
    return monitor.timeit(name)


def log_memory(tag: str = ""):
    """记录内存"""
    monitor.memory(tag)
