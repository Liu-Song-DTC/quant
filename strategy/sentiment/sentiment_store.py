"""情绪时间序列存储 - CSV 持久化，支持回测查询"""
import json
from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd


class SentimentStore:
    """行业情绪分数的时序存储"""

    def __init__(self, data_dir: str = "data/sentiment_data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.processed_dir / "rolling_sentiment.csv"

    COLUMNS = ["date", "industry", "sentiment_score", "importance_avg", "news_count"]

    def save_daily_sentiment(
        self,
        target_date: date,
        scores: Dict[str, float],
        raw_analysis: List[Dict],
    ):
        """保存一天的情绪分数"""
        # 保存 CSV
        rows = []
        importance_by_ind = {}
        count_by_ind = {}
        for r in raw_analysis:
            ind = r["industry"]
            importance_by_ind[ind] = importance_by_ind.get(ind, [])
            importance_by_ind[ind].append(r["importance"])
            count_by_ind[ind] = count_by_ind.get(ind, 0) + 1

        for industry, score in scores.items():
            imps = importance_by_ind.get(industry, [])
            rows.append({
                "date": target_date.isoformat(),
                "industry": industry,
                "sentiment_score": round(score, 4),
                "importance_avg": round(sum(imps) / len(imps), 2) if imps else 0,
                "news_count": count_by_ind.get(industry, 0),
            })

        df = pd.DataFrame(rows)
        if self._csv_path.exists():
            existing = pd.read_csv(self._csv_path, parse_dates=["date"])
            # 移除同日同行业旧数据
            existing = existing[~(
                (existing["date"].dt.date == target_date) &
                (existing["industry"].isin(scores.keys()))
            )]
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(self._csv_path, index=False)

        # 保存原始分析
        raw_path = self.processed_dir / f"{target_date.isoformat()}_analysis.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_analysis, f, ensure_ascii=False, indent=2)

    def load_daily_sentiment(self, target_date: date) -> Dict[str, float]:
        """加载某一天的情绪分数"""
        if not self._csv_path.exists():
            return {}
        df = pd.read_csv(self._csv_path, parse_dates=["date"])
        day_data = df[df["date"].dt.date == target_date]
        if day_data.empty:
            return {}
        return dict(zip(day_data["industry"], day_data["sentiment_score"]))

    def query_range(self, start: date, end: date) -> pd.DataFrame:
        """查询日期范围内的情绪分数，返回矩形 DataFrame（行=日期, 列=行业）"""
        if not self._csv_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(self._csv_path, parse_dates=["date"])
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        df = df[mask]

        if df.empty:
            return pd.DataFrame()

        pivot = df.pivot_table(
            index="date",
            columns="industry",
            values="sentiment_score",
            aggfunc="mean",
        )
        return pivot.sort_index()

    def get_latest_sentiment(self, n_days: int = 3) -> Dict[str, float]:
        """获取最近 n 天的滚动平均情绪"""
        if not self._csv_path.exists():
            return {}

        df = pd.read_csv(self._csv_path, parse_dates=["date"])
        if df.empty:
            return {}

        latest_date = df["date"].max().date()
        start_date = latest_date - timedelta(days=n_days)
        mask = (df["date"].dt.date >= start_date)
        recent = df[mask]

        if recent.empty:
            return {}

        avg = recent.groupby("industry")["sentiment_score"].mean()
        return avg.to_dict()
