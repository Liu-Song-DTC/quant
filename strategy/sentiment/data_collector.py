"""新闻数据采集 - 使用 akshare 抓取每日财经新闻"""
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Optional


class NewsCollector:
    """获取并整理每日金融新闻，供 LLM 分析"""

    def __init__(self, data_dir: str = "data/sentiment_data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def collect_daily_news(self, target_date: Optional[date] = None) -> List[Dict[str, str]]:
        """获取全市场新闻，去重后返回"""
        all_items = []
        all_items.extend(self._fetch_akshare_news())
        all_items.extend(self._fetch_policy_news())
        return self._deduplicate(all_items)

    def _fetch_akshare_news(self) -> List[Dict[str, str]]:
        """从 akshare 获取全市场新闻"""
        try:
            import akshare as ak
            df = ak.stock_news_em()
            if df is None or df.empty:
                return []

            items = []
            for _, row in df.iterrows():
                headline = str(row.get("标题", row.get("title", "")))
                content = str(row.get("内容", row.get("content", "")))
                if not headline or headline == "nan":
                    continue
                items.append({
                    "source": "eastmoney",
                    "headline": headline,
                    "content": content[:500] if content and content != "nan" else headline,
                    "datetime": str(row.get("发布时间", row.get("datetime", ""))),
                })
            return items
        except ImportError:
            return []
        except Exception as e:
            print(f"[NewsCollector] akshare 新闻获取失败: {e}")
            return []

    def _fetch_policy_news(self) -> List[Dict[str, str]]:
        """抓取政策相关新闻（从 akshare 获取宏观/政策栏目）"""
        try:
            import akshare as ak
            df = ak.stock_info_global_em()
            if df is None or df.empty:
                return []

            items = []
            for _, row in df.iterrows():
                title = str(row.get("title", row.get("标题", "")))
                if not title or title == "nan":
                    continue
                content = str(row.get("content", row.get("内容", title)))
                items.append({
                    "source": "global",
                    "headline": title,
                    "content": content[:500] if content and content != "nan" else title,
                    "datetime": str(row.get("time", row.get("datetime", ""))),
                })
            return items
        except ImportError:
            return []
        except Exception as e:
            print(f"[NewsCollector] 全球新闻获取失败: {e}")
            return []

    def _deduplicate(self, items: List[Dict]) -> List[Dict]:
        """基于标题哈希去重"""
        seen = set()
        unique = []
        for item in items:
            h = hashlib.md5(item["headline"][:80].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(item)
        return unique

    def _fetch_futures_data(self) -> List[Dict[str, str]]:
        """获取期货市场数据概况（作为额外的市场信息源）"""
        try:
            import akshare as ak
            df = ak.futures_zh_daily_sina(symbol="RB0")
            if df is None or df.empty:
                return []

            # 取最近几行作为描述
            recent = df.tail(3)
            summary = f"螺纹钢期货近3日: " + ", ".join(
                f"{r['date']} 收盘{r['close']}" for _, r in recent.iterrows()
            )
            return [{
                "source": "futures",
                "headline": "商品期货概况",
                "content": summary,
                "datetime": str(recent.iloc[-1]["date"]) if len(recent) > 0 else "",
            }]
        except Exception:
            return []

    def collect_daily_news_full(self, target_date: Optional[date] = None) -> List[Dict[str, str]]:
        """完整采集：新闻 + 政策 + 期货"""
        items = self.collect_daily_news(target_date)
        items.extend(self._fetch_futures_data())
        return items

    def save_raw(self, items: List[Dict], target_date: date):
        path = self.raw_dir / f"{target_date.isoformat()}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def load_raw(self, target_date: date) -> List[Dict]:
        path = self.raw_dir / f"{target_date.isoformat()}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
