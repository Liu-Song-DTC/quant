"""情绪数据回填 - 基于申万行业指数生成历史行业情绪代理

由于 akshare 新闻 API 不支持历史日期查询，回测期的情绪数据
使用申万行业指数的截面表现作为 LLM 情绪分析的代理指标。

方法：
1. 下载申万一级/二级行业指数历史数据
2. 映射到系统的 14 个行业分类
3. 计算每日行业相对收益 → 滚动 z-score → 压缩到 [-1, 1]
4. 输出为 SentimentStore 兼容的 rolling_sentiment.csv
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import akshare_proxy_patch  # noqa: F401 - 代理补丁
import akshare as ak

ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT / "strategy"))

from core.industry_mapping import get_all_categories

# 申万二级行业指数 → 20 个系统行业分类
SW_TO_CATEGORY: Dict[str, List[str]] = {
    "人工智能/算力": ["801104"],  # 计算机应用（含 AI/云计算）
    "互联网/软件": ["801103", "801206"],
    "通信/计算机": ["801101", "801102", "801223"],
    "半导体/光伏": ["801081", "801735", "801082"],
    "电子": ["801083", "801084", "801085", "801086"],
    "新能源车/风电": ["801095", "801096", "801093", "801092", "801737", "801736", "801731", "801733"],
    "电力设备": ["801161", "801738", "801163"],
    "有色/钢铁/煤炭": [
        "801053", "801054", "801055", "801056", "801051",
        "801043", "801044", "801045", "801951", "801952",
    ],
    "化工": ["801032", "801033", "801034", "801036", "801037", "801038", "801039"],
    "建材": ["801711", "801712", "801713"],
    "军工": ["801741", "801742", "801743", "801744", "801745"],
    "自动化/制造": ["801072", "801074", "801077", "801078"],
    "基建/地产/石油石化": [
        "801721", "801722", "801723", "801724", "801726",
        "801181", "801183", "801962", "801963", "801076",
    ],
    "消费": [
        "801124", "801125", "801126", "801127", "801128", "801129",
        "801111", "801112", "801113", "801114", "801115", "801116",
        "801131", "801132", "801133", "801141", "801142", "801143", "801145",
        "801981", "801982", "801202", "801203", "801204", "801231",
        "801219", "801993", "801994", "801995",
    ],
    "医药": ["801151", "801152", "801153", "801154", "801155", "801156"],
    "传媒": ["801764", "801765", "801766", "801767", "801769"],
    "农业": ["801012", "801014", "801015", "801016", "801017", "801018"],
    "环保/公用": ["801971", "801972"],
    "金融": ["801191", "801193", "801194", "801782", "801783", "801784", "801785"],
    "交运": ["801991", "801992", "801178", "801179"],
}


def download_sw_indices(symbols: List[str], progress: bool = True) -> Dict[str, pd.DataFrame]:
    """下载申万行业指数历史数据，返回 {code: DataFrame}"""
    results = {}
    for code in symbols:
        try:
            df = ak.index_hist_sw(symbol=code, period="day")
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.set_index("日期").sort_index()
            # 计算日收益率（使用收盘价）
            df["ret"] = df["收盘"].pct_change()
            results[code] = df[["ret"]]
            if progress:
                print(f"  ✓ {code}")
        except Exception as e:
            if progress:
                print(f"  ✗ {code}: {e}")
    return results


def build_category_returns(
    sw_data: Dict[str, pd.DataFrame],
    mapping: Dict[str, List[str]],
) -> pd.DataFrame:
    """将申万指数收益率聚合为 14 个行业类别的日收益率

    Returns:
        DataFrame: index=date, columns=14 categories, values=daily return
    """
    category_rets = {}
    for category, codes in mapping.items():
        available = [sw_data[c] for c in codes if c in sw_data]
        if not available:
            print(f"  ⚠ {category}: 无可用指数")
            continue

        # 等权平均各申万子行业收益率
        all_rets = pd.concat([s["ret"] for s in available], axis=1)
        avg_ret = all_rets.mean(axis=1)
        category_rets[category] = avg_ret

    df = pd.DataFrame(category_rets).sort_index()
    return df.dropna(how="all")


def returns_to_sentiment(
    category_rets: pd.DataFrame,
    rolling_window: int = 20,
    half_life: int = 5,
) -> pd.DataFrame:
    """将行业日收益率转换为情绪分数 [-1, 1]

    方法：
    1. EWM 平滑日收益率（捕捉短期行业动能）
    2. 截面 z-score（行业间相对强弱）
    3. tanh 压缩到 [-1, 1]

    无 look-ahead bias：每步只用当日及之前数据。

    Args:
        category_rets: 行业日收益率 DataFrame
        rolling_window: 计算截面 z-score 的滚动窗口
        half_life: EWM 平滑半衰期（天）

    Returns:
        DataFrame: 行业情绪分数 [-1, 1]
    """
    # Step 1: EWM 平滑 → 行业动能
    momentum = category_rets.ewm(halflife=half_life, min_periods=3).mean()

    # Step 2: 滚动截面 z-score（每行 cross-section）
    def cross_section_zscore(row):
        valid = row.dropna()
        if len(valid) < 3:
            return pd.Series(0.0, index=row.index)
        z = (row - valid.mean()) / (valid.std() + 1e-8)
        return z

    z_scores = momentum.apply(cross_section_zscore, axis=1)

    # Step 3: tanh 压缩 → [-1, 1]
    sentiment = np.tanh(z_scores * 0.8)  # 0.8 控制敏感度

    return sentiment.clip(-1.0, 1.0)


def build_sentiment_csv(sentiment: pd.DataFrame, output_path: Path):
    """将情绪 DataFrame 转为 SentimentStore 兼容的 CSV 格式

    CSV columns: date, industry, sentiment_score, importance_avg, news_count
    """
    rows = []
    for date_idx, row in sentiment.iterrows():
        for industry, score in row.items():
            if pd.notna(score):
                rows.append({
                    "date": date_idx.strftime("%Y-%m-%d") if hasattr(date_idx, "strftime") else str(date_idx)[:10],
                    "industry": industry,
                    "sentiment_score": round(float(score), 4),
                    "importance_avg": 3.0,  # 代理数据默认值
                    "news_count": 0,         # 代理数据标记为 0
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "industry"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n情绪 CSV 已保存: {output_path}")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  行业数: {df['industry'].nunique()}")
    print(f"  总记录数: {len(df)}")
    return df


def main():
    output_dir = ROOT / "data" / "sentiment_data" / "processed"
    output_path = output_dir / "rolling_sentiment.csv"

    print("=" * 60)
    print("  行业情绪数据回填（申万指数代理）")
    print("=" * 60)

    # 收集所有需要下载的申万指数代码
    all_sw_codes = sorted(set(
        code for codes in SW_TO_CATEGORY.values() for code in codes
    ))
    print(f"\n需要下载 {len(all_sw_codes)} 个申万行业指数")

    # 下载
    print("\n下载申万行业指数...")
    sw_data = download_sw_indices(all_sw_codes)
    print(f"成功下载: {len(sw_data)} / {len(all_sw_codes)}")

    # 聚合为 14 个行业类别
    print("\n聚合为 14 个行业类别...")
    category_rets = build_category_returns(sw_data, SW_TO_CATEGORY)
    valid_categories = list(category_rets.columns)
    print(f"覆盖行业: {len(valid_categories)} / 14")
    for cat in get_all_categories():
        status = "✓" if cat in valid_categories else "✗"
        print(f"  {status} {cat}")

    # 转换为情绪分数
    print("\n计算情绪分数...")
    sentiment = returns_to_sentiment(category_rets)
    print(f"情绪日期范围: {sentiment.index.min().date()} ~ {sentiment.index.max().date()}")

    # 输出 CSV
    build_sentiment_csv(sentiment, output_path)

    # 验证
    print("\n验证:")
    print(f"  均值: {sentiment.mean().mean():.4f}")
    print(f"  标准差: {sentiment.std().mean():.4f}")
    print(f"  NaN 比例: {sentiment.isna().sum().sum() / sentiment.size:.2%}")


if __name__ == "__main__":
    main()
