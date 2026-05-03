"""情绪分析编排器 - 协调 采集 → 分析 → 存储 → 通知 全流程"""
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .data_collector import NewsCollector
from .llm_analyzer import IndustrySentimentAnalyzer
from .sentiment_store import SentimentStore

ROOT = Path(__file__).parent.parent.parent.resolve()


class SentimentOrchestrator:
    """行业情绪分析的顶层编排器

    支持两种模式：
    - 实盘模式: 采集新闻 + LLM分析 + 存储
    - 回测模式: 从预计算的CSV加载历史情绪数据
    """

    def __init__(self, config, backtest_mode: bool = False):
        """
        Args:
            config: ConfigLoader 实例或 dict
            backtest_mode: True=仅从CSV读取，无需API密钥
        """
        self._cfg = config
        self._backtest_mode = backtest_mode

        data_dir = self._get_config("industry_sentiment.data_dir", "data/sentiment_data")
        if not Path(data_dir).is_absolute():
            data_dir = str(ROOT / data_dir)

        api_key = self._get_config("industry_sentiment.deepseek_api_key", "")
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("ANTHROPIC_AUTH_TOKEN", ""))

        self.collector = None if backtest_mode else NewsCollector(data_dir=data_dir)
        self.store = SentimentStore(data_dir=data_dir)

        # 回测模式：analyzer 仅用于权重计算（纯函数，无需API）
        self.analyzer = IndustrySentimentAnalyzer(api_key=api_key if not backtest_mode else "backtest-mode")

        self.max_news_batch = int(self._get_config("industry_sentiment.max_news_batch", 50))
        self.notify_enabled = self._get_config("industry_sentiment.notify_on_completion", True)

        wi = self._get_config("industry_sentiment.weight_impact", {}) or {}
        self.max_multiplier = float(wi.get("max_multiplier", 1.15))
        self.min_multiplier = float(wi.get("min_multiplier", 0.70))
        self.smoothing_halflife = int(wi.get("smoothing_halflife", 5))
        self.smoothing_method = wi.get("smoothing_method", "ema")
        self.regime_adjustment = wi.get("regime_adjustment", True)
        self.negativity_bias = float(wi.get("negativity_bias", 1.5))

        # 回测模式：加载预计算的情绪数据
        if backtest_mode:
            precomputed = self._get_config("industry_sentiment.backtest.precomputed_file", "")
            if not precomputed:
                precomputed = str(ROOT / "data" / "sentiment_data" / "processed" / "rolling_sentiment.csv")
            if Path(precomputed).exists() and precomputed != str(self.store._csv_path):
                import shutil
                import pandas as pd
                df = pd.read_csv(precomputed)
                df.to_csv(self.store._csv_path, index=False)
                print(f"[Sentiment] 回测模式：已加载预计算情绪数据 {precomputed} ({len(df)} 条)")
            elif self.store._csv_path.exists():
                print(f"[Sentiment] 回测模式：使用已有情绪数据 {self.store._csv_path}")

    def _get_config(self, key: str, default=None):
        """从配置中获取值，兼容 ConfigLoader 和 dict"""
        try:
            return self._cfg.get(key, default)
        except AttributeError:
            return default

    def run_daily(
        self,
        target_date: Optional[date] = None,
        notify: bool = True,
    ) -> Dict[str, float]:
        """执行每日情绪分析：采集 → 分析 → 存储 → 通知"""
        if target_date is None:
            target_date = date.today()

        # Step 1: 采集新闻
        print(f"[Sentiment] 采集 {target_date} 新闻...")
        news = self.collector.collect_daily_news_full(target_date)

        if not news:
            print("[Sentiment] 今日无新闻数据，跳过情绪分析")
            return {}

        print(f"[Sentiment] 采集到 {len(news)} 条新闻")
        self.collector.save_raw(news, target_date)

        # Step 2: LLM 分析
        print(f"[Sentiment] LLM 分析中...")
        scores, raw_analysis = self.analyzer.analyze_day(news)

        if not scores:
            print("[Sentiment] LLM 分析未产生有效结果")
            return {}

        print(f"[Sentiment] 分析完成，覆盖 {len(scores)} 个行业")

        # Step 3: 存储
        self.store.save_daily_sentiment(target_date, scores, raw_analysis)
        print(f"[Sentiment] 结果已保存至 {self.store._csv_path}")

        # Step 4: 打印摘要
        print(f"\n{'='*50}")
        print(f"  行业情绪快报 - {target_date}")
        print(f"{'='*50}")
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        for ind, score in sorted_scores:
            bar = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "⚪")
            print(f"  {bar} {ind:<20s} {score:+.3f}")
        print(f"{'='*50}")

        return scores

    def get_sentiment_weights(self, market_regime: int = 0,
                               current_date=None) -> Dict[str, float]:
        """获取最新的情绪行业权重乘数（EMA 平滑 + 不对称乘数）"""
        scores = self.store.get_latest_sentiment(n_days=self.smoothing_halflife * 2)
        if not scores:
            return {}

        # 回测模式：基于 current_date 查询历史；实盘模式：用最近 N 日
        if current_date is not None:
            ref_date = pd.Timestamp(current_date)
            hist_start = ref_date - pd.Timedelta(days=self.smoothing_halflife * 4)
            sentiment_history = self.store.query_range(
                hist_start.date() if hasattr(hist_start, 'date') else hist_start,
                ref_date.date() if hasattr(ref_date, 'date') else ref_date,
            )
        else:
            sentiment_history = None

        return self.analyzer.get_sentiment_weights_v2(
            scores,
            sentiment_history=sentiment_history if len(sentiment_history) > 0 else None,
            market_regime=market_regime,
            max_multiplier=self.max_multiplier,
            min_multiplier=self.min_multiplier,
            halflife=self.smoothing_halflife,
            negativity_bias=self.negativity_bias,
        )

    def run_backtest(self, start: date, end: date) -> pd.DataFrame:
        """回测模式：处理已持久化的新闻数据（需要预先填充 raw/ 目录）

        遍历 raw/ 目录中已有的新闻文件，对每个日期运行 LLM 分析，
        生成完整的 rolling_sentiment.csv
        """
        raw_files = sorted(self.collector.raw_dir.glob("*.json"))
        for f in raw_files:
            try:
                d = date.fromisoformat(f.stem)
                if start <= d <= end:
                    news = self.collector.load_raw(d)
                    if not news:
                        continue
                    print(f"[Backtest] 分析 {d} - {len(news)} 条新闻")
                    scores, raw_analysis = self.analyzer.analyze_day(news)
                    if scores:
                        self.store.save_daily_sentiment(d, scores, raw_analysis)
            except ValueError:
                continue

        return self.store.query_range(start, end)
