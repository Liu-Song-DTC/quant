"""LLM 行业情绪分析 - 使用 DeepSeek API 批量分析新闻"""
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests

# 确保 industry_mapping 可以被导入
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.industry_mapping import INDUSTRY_KEYWORDS, get_industry_category, get_all_categories

DEEPSEEK_CHAT_URL = "https://api.deepseek.com/v1/chat/completions"


def _build_system_prompt() -> str:
    """构建系统提示，列出 14 个行业及其关键词"""
    industry_desc = []
    for i, (industry, keywords) in enumerate(INDUSTRY_KEYWORDS.items(), 1):
        kw_str = "、".join(keywords[:8])
        industry_desc.append(f"{i}. {industry}（关键词：{kw_str}...）")

    return f"""你是一个中国A股市场行业情绪分析专家。你的任务是从财经新闻中识别涉及的行业，并判断情绪倾向。

## 已定义的14个行业分类：
{chr(10).join(industry_desc)}

## 输出要求：
1. 仅返回 JSON 格式，不要任何额外文字
2. 每条新闻可能涉及多个行业
3. 仅输出你确信相关的行业（有明确提及或强关联）
4. sentiment 取值: "positive"（利好）、"negative"（利空）、"neutral"（中性）
5. importance 取值: 1-5（5=重大政策/事件，1=一般报道）
6. reasoning 简短说明理由（一句话）
7. confidence: 0-1 之间的置信度

## 返回格式：
{{"results": [{{"industry": "行业名", "sentiment": "positive|negative|neutral", "importance": 1-5, "reasoning": "理由", "confidence": 0.8, "related_headlines": ["标题"]}}]}}"""


class IndustrySentimentAnalyzer:
    """使用 DeepSeek API 分析新闻并提取行业情绪"""

    BATCH_SIZE = 50
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(self, api_key: str):
        self.api_key = api_key

    def analyze_day(self, all_news: List[Dict]) -> tuple:
        """分析当天所有新闻，返回 (行业分数, 原始分析结果)"""
        if not all_news:
            return {}, []

        all_results = []
        # 分批处理
        for i in range(0, len(all_news), self.BATCH_SIZE):
            batch = all_news[i:i + self.BATCH_SIZE]
            results = self.analyze_batch(batch)
            if results:
                all_results.extend(results)
            if i + self.BATCH_SIZE < len(all_news):
                time.sleep(0.5)  # 批次间延迟

        scores = self._aggregate_sentiments(all_results)
        return scores, all_results

    def analyze_batch(self, news_items: List[Dict]) -> List[Dict]:
        """分析一批新闻，返回结构化情绪结果"""
        prompt = self._build_prompt(news_items)
        messages = [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(self.MAX_RETRIES):
            raw = self._call_deepseek(messages)
            if raw:
                results = self._parse_response(raw)
                if results:
                    return results
            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_DELAY * (attempt + 1))

        return []

    def _build_prompt(self, news_batch: List[Dict]) -> str:
        """将新闻批次格式化为用户消息"""
        lines = ["请分析以下财经新闻，识别涉及的行业和情绪：", ""]
        for i, item in enumerate(news_batch, 1):
            headline = item.get("headline", "")
            content = item.get("content", "")
            source = item.get("source", "")
            text = content if content and len(content) > len(headline) else headline
            lines.append(f"[{i}] [{source}] {headline}")
            if content and content != headline:
                lines.append(f"    摘要: {content[:300]}")
            lines.append("")

        return "\n".join(lines)

    def _call_deepseek(self, messages: List[Dict]) -> Optional[str]:
        """调用 DeepSeek API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 4096,
                "stream": False,
            }
            resp = requests.post(DEEPSEEK_CHAT_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            print("[LLM] DeepSeek API 超时")
            return None
        except Exception as e:
            print(f"[LLM] DeepSeek API 调用失败: {e}")
            return None

    def _parse_response(self, raw: str) -> List[Dict]:
        """解析 LLM 的 JSON 响应"""
        try:
            # 尝试直接解析
            data = json.loads(raw)
            return self._validate_results(data.get("results", []))
        except json.JSONDecodeError:
            pass

        # 尝试提取 markdown 代码块中的 JSON
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            try:
                data = json.loads(match.group(1))
                return self._validate_results(data.get("results", []))
            except (json.JSONDecodeError, AttributeError):
                pass

        # 尝试找到最外层的 JSON 对象
        match = re.search(r"\{[\s\S]*\"results\"[\s\S]*\}", raw)
        if match:
            try:
                data = json.loads(match.group(0))
                return self._validate_results(data.get("results", []))
            except (json.JSONDecodeError, AttributeError):
                pass

        return []

    def _validate_results(self, results: List[Dict]) -> List[Dict]:
        """验证并清洗 LLM 输出"""
        validated = []
        for r in results:
            industry_name = r.get("industry", "")
            if not industry_name:
                continue

            # 映射到 14 个标准行业
            category = get_industry_category(industry_name)
            if category == "default":
                continue

            sentiment = r.get("sentiment", "neutral")
            if sentiment not in ("positive", "negative", "neutral"):
                sentiment = "neutral"

            importance = max(1, min(5, int(r.get("importance", 3))))
            confidence = max(0.0, min(1.0, float(r.get("confidence", 0.5))))

            validated.append({
                "industry": category,
                "original_industry": industry_name,
                "sentiment": sentiment,
                "importance": importance,
                "confidence": confidence,
                "reasoning": r.get("reasoning", ""),
                "related_headlines": r.get("related_headlines", []),
            })
        return validated

    def _aggregate_sentiments(self, all_results: List[Dict]) -> Dict[str, float]:
        """将多条目情绪汇总为每日行业分数 [-1.0, 1.0]"""
        if not all_results:
            return {}

        # 按行业汇总加权分数
        industry_scores = {}
        for r in all_results:
            ind = r["industry"]
            if ind not in industry_scores:
                industry_scores[ind] = {"weighted_sum": 0.0, "weight_sum": 0.0}

            sign = {"positive": 1, "negative": -1, "neutral": 0}[r["sentiment"]]
            weight = r["importance"] * r["confidence"]
            industry_scores[ind]["weighted_sum"] += sign * weight
            industry_scores[ind]["weight_sum"] += weight

        # 归一化为 [-1.0, 1.0]
        final_scores = {}
        for ind, scores in industry_scores.items():
            if scores["weight_sum"] > 0:
                final_scores[ind] = scores["weighted_sum"] / scores["weight_sum"]
            else:
                final_scores[ind] = 0.0

        return final_scores

    def get_sentiment_weights(
        self,
        latest_scores: Dict[str, float],
        market_regime: int = 0,
        max_multiplier: float = 1.20,
        min_multiplier: float = 0.80,
        regime_adjust: bool = True,
    ) -> Dict[str, float]:
        """将情绪分数转换为行业权重乘数 [min_multiplier, max_multiplier]

        market_regime: 1=bull, 0=neutral, -1=bear
        """
        multipliers = {}
        for industry in get_all_categories():
            score = latest_scores.get(industry, 0.0)
            # score [-1, 1] → multiplier [min_multiplier, max_multiplier]
            multiplier = 1.0 + score * (max_multiplier - 1.0) if score > 0 else 1.0 + score * (1.0 - min_multiplier)

            if regime_adjust:
                if market_regime == -1:  # 熊市：降低正面情绪的影响
                    multiplier = 1.0 + (multiplier - 1.0) * 0.5
                elif market_regime == 1:  # 牛市：放大正面情绪
                    multiplier = 1.0 + (multiplier - 1.0) * 1.2

            multiplier = max(min_multiplier, min(max_multiplier, multiplier))
            multipliers[industry] = multiplier

        return multipliers

    def get_sentiment_weights_v2(
        self,
        latest_scores: Dict[str, float],
        sentiment_history: Optional[pd.DataFrame] = None,
        market_regime: int = 0,
        max_multiplier: float = 1.15,
        min_multiplier: float = 0.70,
        halflife: int = 5,
        negativity_bias: float = 1.5,
    ) -> Dict[str, float]:
        """情绪权重 v2：EMA 平滑 + 不对称乘数

        改进：
        1. EMA 替代 SMA：对近期情绪更高权重，旧情绪自然衰减
        2. 正/负面不对称：负面情绪惩罚更重（negativity_bias 控制幅度）
        3. 熊市负面情绪加倍：熊市中负面情绪乘数放大

        Args:
            latest_scores: 当日情绪分数 {industry: score}
            sentiment_history: 历史情绪 DataFrame（行=日期，列=行业）
            market_regime: 1=bull, 0=neutral, -1=bear
            halflife: EMA 半衰期（天）
            negativity_bias: 负面情绪放大倍数
        """
        multipliers = {}
        for industry in get_all_categories():
            raw_score = latest_scores.get(industry, 0.0)

            # EMA 平滑：从历史序列计算
            if sentiment_history is not None and industry in sentiment_history.columns:
                hist = sentiment_history[industry].dropna()
                if len(hist) > 0:
                    smoothed = float(hist.ewm(halflife=halflife).mean().iloc[-1])
                    score = 0.7 * smoothed + 0.3 * raw_score
                else:
                    score = raw_score
            else:
                score = raw_score

            # 不对称乘数映射
            if score > 0:
                # 正面情绪：[1.0, max_multiplier]，影响较小
                multiplier = 1.0 + score * (max_multiplier - 1.0)
            else:
                # 负面情绪：[min_multiplier, 1.0]，影响更大（negativity_bias）
                multiplier = 1.0 + score * (1.0 - min_multiplier) * negativity_bias

            # 熊市中负面情绪加倍影响
            if market_regime == -1 and score < 0:
                multiplier = 1.0 + (multiplier - 1.0) * 1.5

            multiplier = max(min_multiplier, min(max_multiplier, multiplier))
            multipliers[industry] = multiplier

        return multipliers
