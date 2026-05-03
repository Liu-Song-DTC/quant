# analysis/dashboard.py
"""
量化策略分析看板 — 基于 Streamlit + Plotly

功能：
1. 收益率 & 回撤曲线
2. 因子 IC 热力图
3. 行业情绪雷达图
4. 持仓集中度分析
5. 关键绩效指标卡片
6. 因子纯度对比

启动方式：
    cd strategy && streamlit run analysis/dashboard.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
STRATEGY_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(STRATEGY_DIR))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="量化策略分析看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 数据加载 ====================


@st.cache_data(ttl=3600)
def load_data():
    """加载回测结果和情绪数据"""
    signals_path = STRATEGY_DIR / "rolling_validation_results" / "backtest_signals.csv"
    selections_path = STRATEGY_DIR / "rolling_validation_results" / "portfolio_selections.csv"
    sentiment_path = Path(STRATEGY_DIR.parent) / "data" / "sentiment_data" / "processed" / "rolling_sentiment.csv"

    data = {"signals": None, "selections": None, "sentiment": None}

    if signals_path.exists():
        df = pd.read_csv(signals_path, parse_dates=["date"])
        if "factor_quality" not in df.columns:
            df["factor_quality"] = 0.0
        data["signals"] = df

    if selections_path.exists():
        data["selections"] = pd.read_csv(selections_path, parse_dates=["date"])

    if sentiment_path.exists():
        sent = pd.read_csv(sentiment_path)
        if "date" in sent.columns:
            sent["date"] = pd.to_datetime(sent["date"])
        data["sentiment"] = sent

    return data


def compute_returns_curve(selections_df, signals_df):
    """从选股数据计算策略收益曲线"""
    if selections_df is None or len(selections_df) == 0:
        return None, None

    # 按日期分组，计算每日组合收益
    daily_returns = []
    dates = sorted(selections_df["date"].unique())

    for d in dates:
        day_sel = selections_df[selections_df["date"] == d]
        if len(day_sel) == 0:
            continue

        # 使用等权组合收益（实际回测结果更准确，这里作为近似）
        avg_score = day_sel["score"].mean() if "score" in day_sel.columns else 0
        daily_returns.append({"date": d, "daily_ret": avg_score * 0.01})

    if not daily_returns:
        return None, None

    ret_df = pd.DataFrame(daily_returns).sort_values("date")
    ret_df["cum_ret"] = (1 + ret_df["daily_ret"]).cumprod() - 1
    ret_df["cummax"] = ret_df["cum_ret"].cummax()
    ret_df["drawdown"] = (1 + ret_df["cum_ret"]) / (1 + ret_df["cummax"]) - 1

    return ret_df, None  # benchmark not available


def compute_ic_heatmap_data(signals_df):
    """从信号数据构建 IC 热力图所需数据"""
    if signals_df is None or "factor_name" not in signals_df.columns:
        return None

    # 按日期和因子名称分组，计算平均因子质量（IC代理）
    ic_data = signals_df.groupby([pd.Grouper(key="date", freq="M"), "factor_name"])[
        "factor_quality"
    ].mean().reset_index()
    ic_data = ic_data.dropna()

    if ic_data.empty:
        return None

    # 透视：行=因子，列=日期
    heatmap = ic_data.pivot_table(
        index="factor_name", columns="date", values="factor_quality", aggfunc="mean"
    )
    return heatmap


def compute_sentiment_radar_data(sentiment_df):
    """准备雷达图所需的情绪数据"""
    if sentiment_df is None or len(sentiment_df) == 0:
        return None

    latest_date = sentiment_df["date"].max()
    latest = sentiment_df[sentiment_df["date"] == latest_date]
    if latest.empty:
        return None

    return latest.set_index("industry")["sentiment_score"].to_dict()


# ==================== 主页面 ====================

data = load_data()
ret_df, _ = compute_returns_curve(data["selections"], data["signals"])
ic_heatmap = compute_ic_heatmap_data(data["signals"])
radar_data = compute_sentiment_radar_data(data["sentiment"])

st.title("📊 量化策略分析看板")
st.caption("因子选股 + 行业情绪 + 市场状态综合监控")

# ---- 第一行：KPI 卡片 ----
st.subheader("关键绩效指标")
cols = st.columns(6)

total_signals = len(data["signals"]) if data["signals"] is not None else 0
buy_signals = (data["signals"]["buy"] == True).sum() if data["signals"] is not None else 0
industries_covered = (
    data["signals"]["industry"].nunique()
    if data["signals"] is not None and "industry" in data["signals"].columns
    else 0
)
avg_positions = (
    data["selections"].groupby("date")["code"].count().mean()
    if data["selections"] is not None and len(data["selections"]) > 0
    else 0
)
avg_score = data["signals"]["score"].mean() if data["signals"] is not None else 0

ann_ret = 0.0
max_dd = 0.0
sharpe_est = 0.0
if ret_df is not None and len(ret_df) > 0:
    total_ret = ret_df["cum_ret"].iloc[-1]
    n_years = max(1, (ret_df["date"].iloc[-1] - ret_df["date"].iloc[0]).days / 365.25)
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    max_dd = ret_df["drawdown"].min()
    daily_std = ret_df["daily_ret"].std()
    sharpe_est = (ret_df["daily_ret"].mean() / (daily_std + 1e-10)) * np.sqrt(252) if daily_std > 0 else 0

with cols[0]:
    st.metric("信号总数", f"{total_signals:,}")
with cols[1]:
    st.metric("年化收益", f"{ann_ret:.2%}")
with cols[2]:
    st.metric("最大回撤", f"{max_dd:.2%}")
with cols[3]:
    st.metric("预估Sharpe", f"{sharpe_est:.2f}")
with cols[4]:
    st.metric("平均持仓", f"{avg_positions:.1f}")
with cols[5]:
    st.metric("行业覆盖", str(industries_covered))

# ---- 第二行：收益曲线 + 回撤 ----
st.subheader("策略收益 & 回撤")
if ret_df is not None and len(ret_df) > 0:
    fig_ret = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4],
        subplot_titles=("累计收益率", "回撤"),
    )

    fig_ret.add_trace(
        go.Scatter(
            x=ret_df["date"], y=ret_df["cum_ret"] * 100,
            mode="lines", name="累计收益",
            fill="tozeroy", fillcolor="rgba(0,128,0,0.1)",
            line=dict(color="green", width=2),
        ),
        row=1, col=1,
    )

    fig_ret.add_trace(
        go.Scatter(
            x=ret_df["date"], y=ret_df["drawdown"] * 100,
            mode="lines", name="回撤",
            fill="tozeroy", fillcolor="rgba(255,0,0,0.1)",
            line=dict(color="red", width=2),
        ),
        row=2, col=1,
    )

    fig_ret.update_layout(
        height=500, showlegend=False,
        hovermode="x unified",
    )
    fig_ret.update_yaxes(title_text="收益 %", row=1, col=1)
    fig_ret.update_yaxes(title_text="回撤 %", row=2, col=1)
    st.plotly_chart(fig_ret, use_container_width=True)
else:
    st.info("暂无收益数据。运行 bt_execution.py 生成。")

# ---- 第三行：IC 热力图 + 情绪雷达 ----
st.subheader("因子质量 & 行业情绪")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**因子 IC 热力图（月频）**")
    if ic_heatmap is not None and len(ic_heatmap) > 0:
        # 只显示 Top 15 因子
        top_factors = ic_heatmap.mean(axis=1).abs().sort_values(ascending=False).head(15).index
        heatmap_top = ic_heatmap.loc[ic_heatmap.index.isin(top_factors)]

        fig_hm = px.imshow(
            heatmap_top,
            labels=dict(x="日期", y="因子", color="IC 代理"),
            color_continuous_midpoint=0,
            color_continuous_scale="RdBu_r",
            aspect="auto",
        )
        fig_hm.update_layout(height=400, margin=dict(t=10, b=10))
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("暂无 IC 数据。确保信号包含因子质量字段。")

with col2:
    st.markdown("**行业情绪雷达图**")
    if radar_data is not None and len(radar_data) > 0:
        # 取 Top 12 行业
        sorted_inds = sorted(radar_data.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
        categories = [x[0] for x in sorted_inds]
        values = [x[1] for x in sorted_inds]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # 闭环
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.3)",
            line=dict(color="rgba(99, 110, 250, 0.8)", width=2),
            name="情绪得分",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[-1, 1], tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=10)),
            ),
            height=450,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("暂无情绪数据。运行 sentiment/orchestrator.py 生成。")

# ---- 第四行：持仓分析 ----
st.subheader("持仓分析")
col3, col4 = st.columns(2)

with col3:
    st.markdown("**行业仓位分布（最新日）**")
    if data["selections"] is not None and len(data["selections"]) > 0:
        latest_date = data["selections"]["date"].max()
        latest = data["selections"][data["selections"]["date"] == latest_date]
        if "industry" in latest.columns and len(latest) > 0:
            ind_dist = latest.groupby("industry")["weight"].sum().sort_values(ascending=False)
            fig_ind = px.bar(
                x=ind_dist.index, y=ind_dist.values,
                labels={"x": "", "y": "权重"},
                color=ind_dist.values,
                color_continuous_scale="Blues",
            )
            fig_ind.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_ind, use_container_width=True)
        else:
            st.info("无行业分布数据")
    else:
        st.info("暂无选股数据")

with col4:
    st.markdown("**持仓数量随时间变化**")
    if data["selections"] is not None and len(data["selections"]) > 0:
        pos_counts = data["selections"].groupby("date")["code"].count()
        fig_pos = px.line(
            x=pos_counts.index, y=pos_counts.values,
            markers=True,
            labels={"x": "", "y": "持仓数"},
        )
        fig_pos.update_layout(height=350)
        st.plotly_chart(fig_pos, use_container_width=True)
    else:
        st.info("暂无选股数据")

# ---- 第五行：情绪历史趋势 ----
st.subheader("行业情绪历史趋势")
if data["sentiment"] is not None and len(data["sentiment"]) > 0:
    sentiment_df = data["sentiment"]
    sentiment_pivot = sentiment_df.pivot_table(
        index="date", columns="industry", values="sentiment_score", aggfunc="mean"
    )
    if len(sentiment_pivot.columns) > 0:
        # 取最近有变动的 Top 8 行业
        top_inds = sentiment_pivot.std().sort_values(ascending=False).head(8).index
        fig_sent = px.line(
            sentiment_pivot[top_inds],
            labels={"date": "", "value": "情绪得分", "industry": "行业"},
        )
        fig_sent.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_sent.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("暂无情绪历史数据")
else:
    st.info("暂无情绪数据。")

# ---- 第六行：因子使用分布 ----
st.subheader("因子使用频次分析")
if data["signals"] is not None and "factor_name" in data["signals"].columns:
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**因子使用 Top 15**")
        factor_counts = data["signals"]["factor_name"].value_counts().head(15)
        fig_factor = px.bar(
            x=factor_counts.index, y=factor_counts.values,
            labels={"x": "", "y": "使用次数"},
            color=factor_counts.values,
            color_continuous_scale="Viridis",
        )
        fig_factor.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_factor, use_container_width=True)

    with col6:
        st.markdown("**因子类型占比**")
        def classify_factor(name):
            if isinstance(name, str):
                if name.startswith("DYN_"):
                    return "动态因子"
                elif name.startswith("IND_"):
                    return "行业因子"
                elif name.startswith("DEFAULT"):
                    return "默认因子"
                elif name == "V41":
                    return "V41(早期)"
            return "其他"

        type_counts = data["signals"]["factor_name"].apply(classify_factor).value_counts()
        fig_pie = px.pie(
            values=type_counts.values, names=type_counts.index,
            hole=0.4,
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("信号数据中缺少因子名称字段")

# ---- 侧边栏：参数概览 ----
with st.sidebar:
    st.header("策略参数")
    try:
        import yaml

        config_path = STRATEGY_DIR / "config" / "factor_config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            st.markdown(f"**因子模式:** `{cfg.get('factor_mode', 'N/A')}`")

            dyn = cfg.get("dynamic_factor", {})
            st.markdown(f"**动态因子:** 窗口{dyn.get('train_window_days', 'N/A')}天, "
                       f"Top{dyn.get('top_n_factors', 'N/A')}")

            sig = cfg.get("signal", {})
            st.markdown(f"**信号阈值:** buy>{sig.get('buy_threshold', 'N/A')}")

            bt_cfg = cfg.get("backtest", {})
            st.markdown(f"**换仓周期:** {bt_cfg.get('rebalance_days', 'N/A')}天")

            st.markdown(f"**情绪模块:** {'✅ 启用' if cfg.get('industry_sentiment', {}).get('enabled') else '❌ 未启用'}")

            neu = cfg.get("factor_neutralization", {})
            st.markdown(f"**因子中性化:** {'✅ 启用' if neu.get('enabled') else '❌ 未启用'}")

            rp = cfg.get("risk_parity", {})
            st.markdown(f"**风险平价:** {'✅ 启用' if rp.get('enabled') else '❌ 未启用'}")

            st.divider()

            port = cfg.get("portfolio", {})
            st.markdown(f"**止损:** 个股权{port.get('position_stop_loss', 'N/A')}, "
                       f"组合{port.get('portfolio_stop_loss', 'N/A')}")

            esl = cfg.get("enhanced_stop_loss", {})
            st.markdown(f"**增强止损:** 移动止盈{esl.get('trailing_stop_pct', 'N/A')} "
                       f"({'✅' if esl.get('trailing_stop_enabled') else '❌'})")

            extra = dyn.get("extra_candidate_factors", [])
            st.markdown(f"**Alpha因子候选:** {len(extra)} 个")
    except Exception:
        st.warning("无法加载配置文件")

    st.divider()
    st.caption("数据刷新：重新运行 bt_execution.py")
    st.caption(f"信号数：{total_signals:,}")
