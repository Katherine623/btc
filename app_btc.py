# app_btc.py
# -*- coding: utf-8 -*-
"""
Bitcoin RL Trading (PPO) — Streamlit 互動介面
執行方式：streamlit run app_btc.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from btc_rl_trading_ppo import (
    download_btc_data,
    load_data,
    add_technical_indicators,
    build_feature_columns,
    BitcoinTradingEnv,
    evaluate_agent,
    compute_metrics,
)

# ──────────────────────────────────────────────
# 頁面設定
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bitcoin RL 交易模型",
    page_icon="₿",
    layout="wide",
)

st.title("₿ 比特幣強化學習交易模型（PPO）")
st.markdown(
    "使用 **Proximal Policy Optimization (PPO)** 訓練 BTC 交易代理人，"
    "自動學習買入 / 賣出 / 持倉策略。"
)

# ──────────────────────────────────────────────
# 側邊欄：資料 & 超參數
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 參數設定")

    st.subheader("資料來源")
    data_source = st.radio(
        "選擇資料方式",
        ["從 yfinance 下載", "上傳 CSV 檔案"],
        index=0,
    )

    yf_period = st.selectbox(
        "下載期間",
        ["1y", "2y", "5y", "max"],
        index=1,
    )
    yf_interval = st.selectbox(
        "K 棒週期",
        ["1d", "1h"],
        index=0,
    )

    uploaded_file = None
    if data_source == "上傳 CSV 檔案":
        uploaded_file = st.file_uploader("上傳 CSV（需含 Open/High/Low/Close/Volume）", type=["csv"])

    st.divider()

    st.subheader("訓練超參數")
    initial_balance = st.number_input("初始資金（USD）", value=10000, min_value=100, step=500)
    trade_fee = st.slider("交易手續費", min_value=0.0, max_value=0.01, value=0.001, step=0.0005, format="%.4f")
    total_timesteps = st.select_slider(
        "訓練總步數",
        options=[50_000, 100_000, 200_000, 300_000, 500_000],
        value=200_000,
    )
    train_split = st.slider("訓練集比例", min_value=0.5, max_value=0.9, value=0.8, step=0.05)

    st.divider()
    st.subheader("⚡ 執行速度")
    performance_mode = st.radio(
        "執行模式",
        ["快速模式", "完整模式"],
        index=0,
        horizontal=True,
    )
    use_saved_model = st.checkbox("優先載入既有模型（若存在）", value=True)
    run_stress_test = st.checkbox("啟用成本壓力測試", value=(performance_mode == "完整模式"))
    if performance_mode == "快速模式":
        fast_max_bars = st.number_input("快速模式最大資料筆數", min_value=300, max_value=3000, value=900, step=100)
    else:
        fast_max_bars = 3000

    st.divider()

    # Session defaults for auto/advanced configuration.
    defaults = {
        "slippage_bps": 8.0,
        "spread_bps": 4.0,
        "maker_fee": 0.0002,
        "taker_fee": 0.0007,
        "min_trade_pct": 0.02,
        "min_notional": 10.0,
        "min_qty": 0.0001,
        "qty_step": 0.0001,
        "price_step": 0.01,
        "position_step": 0.25,
        "slippage_vol_multiplier": 1.2,
        "max_drawdown_limit": 0.30,
        "daily_loss_limit": 0.06,
        "volatility_target": 0.02,
        "wf_train_window": 360,
        "wf_test_window": 120,
        "wf_max_folds": 5,
        "wf_timesteps": 20_000,
        "enable_walk_forward": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "base_threshold" not in st.session_state:
        st.session_state.base_threshold = 0.55
    if "strictness_multiplier" not in st.session_state:
        st.session_state.strictness_multiplier = 1.15
    if "strategy_preset" not in st.session_state:
        st.session_state.strategy_preset = "平衡"

    ui_mode = st.radio(
        "參數模式",
        ["自動模式", "進階模式"],
        index=0,
        horizontal=True,
    )

    st.subheader("🎛️ 策略風格")
    st.caption("一鍵預設（保守 / 平衡 / 積極）")
    p1, p2, p3 = st.columns(3)
    if p1.button("保守", use_container_width=True):
        st.session_state.base_threshold = 0.62
        st.session_state.strictness_multiplier = 1.30
        st.session_state.min_trade_pct = 0.03
        st.session_state.position_step = 0.20
        st.session_state.enable_walk_forward = False
        st.session_state.strategy_preset = "保守"
    if p2.button("平衡", use_container_width=True):
        st.session_state.base_threshold = 0.55
        st.session_state.strictness_multiplier = 1.15
        st.session_state.min_trade_pct = 0.02
        st.session_state.position_step = 0.25
        st.session_state.enable_walk_forward = False
        st.session_state.strategy_preset = "平衡"
    if p3.button("積極", use_container_width=True):
        st.session_state.base_threshold = 0.48
        st.session_state.strictness_multiplier = 1.05
        st.session_state.min_trade_pct = 0.01
        st.session_state.position_step = 0.33
        st.session_state.enable_walk_forward = False
        st.session_state.strategy_preset = "積極"

    if ui_mode == "自動模式":
        st.info(
            f"目前使用 {st.session_state.strategy_preset} 預設："
            f"Regime 門檻 {st.session_state.base_threshold:.2f} / "
            f"高波動倍數 {st.session_state.strictness_multiplier:.2f} / "
            f"調倉步長 {st.session_state.position_step:.2f}"
        )
    else:
        st.divider()
        st.subheader("🏦 現實市場設定")
        st.session_state.slippage_bps = st.slider("基礎滑價（bps）", min_value=0.0, max_value=30.0, value=float(st.session_state.slippage_bps), step=1.0)
        st.session_state.spread_bps = st.slider("買賣價差（bps）", min_value=0.0, max_value=20.0, value=float(st.session_state.spread_bps), step=1.0)
        st.session_state.maker_fee = st.slider("Maker 費率", min_value=0.0, max_value=0.0020, value=float(st.session_state.maker_fee), step=0.0001, format="%.4f")
        st.session_state.taker_fee = st.slider("Taker 費率", min_value=0.0, max_value=0.0030, value=float(st.session_state.taker_fee), step=0.0001, format="%.4f")
        st.session_state.min_trade_pct = st.slider("最小成交比例", min_value=0.0, max_value=0.10, value=float(st.session_state.min_trade_pct), step=0.005)
        st.session_state.min_notional = st.number_input("最小名目金額（USD）", min_value=1.0, value=float(st.session_state.min_notional), step=1.0)
        st.session_state.min_qty = st.number_input("最小下單數量（BTC）", min_value=0.00001, value=float(st.session_state.min_qty), step=0.00001, format="%.5f")
        st.session_state.qty_step = st.number_input("下單數量精度步進", min_value=0.00001, value=float(st.session_state.qty_step), step=0.00001, format="%.5f")
        st.session_state.price_step = st.number_input("價格精度步進", min_value=0.01, value=float(st.session_state.price_step), step=0.01)
        st.session_state.position_step = st.select_slider("單次調倉步長", options=[0.10, 0.20, 0.25, 0.33, 0.50], value=float(st.session_state.position_step))
        st.session_state.slippage_vol_multiplier = st.slider("高波動滑價放大倍數", min_value=0.0, max_value=3.0, value=float(st.session_state.slippage_vol_multiplier), step=0.1)
        st.session_state.volatility_target = st.slider("波動目標（風險縮放）", min_value=0.005, max_value=0.05, value=float(st.session_state.volatility_target), step=0.001, format="%.3f")

        st.divider()
        st.subheader("🛑 風險引擎")
        st.session_state.max_drawdown_limit = st.slider("最大回撤停機線", min_value=0.10, max_value=0.60, value=float(st.session_state.max_drawdown_limit), step=0.01)
        st.session_state.daily_loss_limit = st.slider("單次回測虧損停機線", min_value=0.02, max_value=0.30, value=float(st.session_state.daily_loss_limit), step=0.01)

        st.divider()
        st.subheader("🧪 Walk-forward 回測")
        st.session_state.enable_walk_forward = st.checkbox("啟用 Walk-forward 滾動回測", value=bool(st.session_state.enable_walk_forward))
        if st.session_state.enable_walk_forward:
            st.session_state.wf_train_window = int(st.number_input("每折訓練長度（bars）", min_value=120, value=int(st.session_state.wf_train_window), step=60))
            st.session_state.wf_test_window = int(st.number_input("每折測試長度（bars）", min_value=48, value=int(st.session_state.wf_test_window), step=24))
            st.session_state.wf_max_folds = int(st.number_input("最多折數", min_value=2, max_value=12, value=int(st.session_state.wf_max_folds), step=1))
            st.session_state.wf_timesteps = int(
                st.select_slider(
                    "每折訓練步數",
                    options=[10_000, 20_000, 30_000, 50_000, 80_000],
                    value=int(st.session_state.wf_timesteps),
                )
            )

        st.divider()
        st.subheader("🧠 Regime 門檻控制")
        st.session_state.base_threshold = st.slider(
            "基準信心閾值",
            min_value=0.45,
            max_value=0.75,
            value=float(st.session_state.base_threshold),
            step=0.01,
        )
        st.session_state.strictness_multiplier = st.slider(
            "高波動嚴格倍數",
            min_value=1.0,
            max_value=1.4,
            value=float(st.session_state.strictness_multiplier),
            step=0.05,
        )

    slippage_bps = float(st.session_state.slippage_bps)
    spread_bps = float(st.session_state.spread_bps)
    maker_fee = float(st.session_state.maker_fee)
    taker_fee = float(st.session_state.taker_fee)
    min_trade_pct = float(st.session_state.min_trade_pct)
    min_notional = float(st.session_state.min_notional)
    min_qty = float(st.session_state.min_qty)
    qty_step = float(st.session_state.qty_step)
    price_step = float(st.session_state.price_step)
    position_step = float(st.session_state.position_step)
    slippage_vol_multiplier = float(st.session_state.slippage_vol_multiplier)
    max_drawdown_limit = float(st.session_state.max_drawdown_limit)
    daily_loss_limit = float(st.session_state.daily_loss_limit)
    volatility_target = float(st.session_state.volatility_target)

    enable_walk_forward = bool(st.session_state.enable_walk_forward)
    wf_train_window = int(st.session_state.wf_train_window)
    wf_test_window = int(st.session_state.wf_test_window)
    wf_max_folds = int(st.session_state.wf_max_folds)
    wf_timesteps = int(st.session_state.wf_timesteps)
    base_threshold = float(st.session_state.base_threshold)
    strictness_multiplier = float(st.session_state.strictness_multiplier)

    st.divider()
    run_btn = st.button("🚀 開始下載 & 訓練", use_container_width=True)

# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
CSV_PATH = "btc_usdt_1h.csv"
MODEL_PATH = "ppo_btc_trading_agent"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yfinance_cached(interval: str, period: str, csv_path: str) -> pd.DataFrame:
    return download_btc_data(
        symbol="BTC-USD",
        interval=interval,
        period=period,
        save_path=csv_path,
        max_retries=3,
    )


@st.cache_data(show_spinner=False)
def add_technical_indicators_cached(df: pd.DataFrame) -> pd.DataFrame:
    return add_technical_indicators(df)


def load_or_download_data() -> pd.DataFrame:
    if data_source == "上傳 CSV 檔案" and uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ CSV 讀取失敗：{str(e)}")
            st.stop()
    elif data_source == "從 yfinance 下載":
        try:
            with st.spinner("正在從 yfinance 下載 BTC 資料...（可能需要 10-30 秒）"):
                df_raw = fetch_yfinance_cached(yf_interval, yf_period, CSV_PATH)
        except Exception as e:
            st.error(
                f"❌ yfinance 下載失敗。\n\n"
                f"**原因：** {str(e)}\n\n"
                f"**建議：**\n"
                f"1. 等待 1-2 分鐘後再試（API 限流）\n"
                f"2. 改用「上傳 CSV 檔案」方式\n"
                f"3. 使用更短的時間週期（例如 1y 而非 max）"
            )
            st.stop()
    elif os.path.exists(CSV_PATH):
        df_raw = pd.read_csv(CSV_PATH)
        st.info(f"✓ 讀取已存在的資料檔：{CSV_PATH}（{len(df_raw)} 筆）")
    else:
        st.error("❌ 請選擇資料來源或上傳 CSV 後再執行。")
        st.stop()

    return df_raw


def get_time_axis(df: pd.DataFrame, length: int):
    if "Datetime" in df.columns:
        dt = pd.to_datetime(df["Datetime"], errors="coerce").iloc[:length]
        if dt.notna().all() and len(dt) == length:
            return dt, True
    return pd.RangeIndex(start=0, stop=length, step=1), False


def plot_equity_curve(equity_curve, buy_hold_curve, time_axis, use_datetime, title="Equity Curve"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, equity_curve, label="RL Agent", color="#f7931a", linewidth=1.5)
    ax.plot(
        time_axis,
        buy_hold_curve[: len(equity_curve)],
        label="Buy & Hold",
        color="#4c72b0",
        linewidth=1.5,
        linestyle="--",
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Datetime" if use_datetime else "Time Step")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if use_datetime:
        fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_action_distribution(action_history):
    labels = ["Hold (0)", "Buy (1)", "Sell (2)"]
    counts = [action_history.count(i) for i in range(3)]
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(labels, counts, color=["#aec6e8", "#77c77a", "#f28b82"])
    ax.set_title("Action Distribution")
    ax.set_ylabel("Count")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(cnt), ha="center", va="bottom")
    plt.tight_layout()
    return fig


def plot_price_with_signals(test_df, action_history, time_axis, use_datetime):
    prices = test_df["Close"].values[: len(action_history)]
    x = np.array(time_axis[: len(action_history)])

    buy_steps  = [i for i, a in enumerate(action_history) if a == 1]
    sell_steps = [i for i, a in enumerate(action_history) if a == 2]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, prices, color="gray", linewidth=1, label="Close Price")
    ax.scatter([x[i] for i in buy_steps],  [prices[i] for i in buy_steps],  marker="^", color="green",  s=60, zorder=5, label="Buy")
    ax.scatter([x[i] for i in sell_steps], [prices[i] for i in sell_steps], marker="v", color="red",    s=60, zorder=5, label="Sell")
    ax.set_title("Price Chart with Trade Signals")
    ax.set_xlabel("Datetime" if use_datetime else "Time Step")
    ax.set_ylabel("BTC Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if use_datetime:
        fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_price_with_regime_overlay(test_df, action_history, time_axis, use_datetime):
    prices = test_df["Close"].values[: len(action_history)]
    x = np.array(time_axis[: len(action_history)])
    regimes = test_df["market_regime"].values[: len(action_history)] if "market_regime" in test_df.columns else None

    buy_steps = [i for i, a in enumerate(action_history) if a == 1]
    sell_steps = [i for i, a in enumerate(action_history) if a == 2]

    fig, ax = plt.subplots(figsize=(10, 4))

    if regimes is not None and len(regimes) > 0:
        regime_colors = {
            "bull_trend": "#dff5e1",
            "bear_trend": "#f9e0e0",
            "range_bound": "#eef2f6",
            "high_volatility": "#fff4d6",
        }
        start = 0
        for i in range(1, len(regimes) + 1):
            if i == len(regimes) or regimes[i] != regimes[start]:
                color = regime_colors.get(regimes[start], "#f3f3f3")
                ax.axvspan(x[start], x[i - 1], color=color, alpha=0.35)
                start = i

    ax.plot(x, prices, color="gray", linewidth=1, label="Close Price")
    ax.scatter([x[i] for i in buy_steps], [prices[i] for i in buy_steps], marker="^", color="green", s=60, zorder=5, label="Buy")
    ax.scatter([x[i] for i in sell_steps], [prices[i] for i in sell_steps], marker="v", color="red", s=60, zorder=5, label="Sell")
    ax.set_title("Price Chart with Regime Overlay")
    ax.set_xlabel("Datetime" if use_datetime else "Time Step")
    ax.set_ylabel("BTC Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if use_datetime:
        fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def compute_advanced_metrics(equity_curve, action_history):
    equity = np.array(equity_curve, dtype=np.float64)
    if len(equity) < 3:
        return {
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "trade_count": 0,
            "trade_density": 0.0,
        }

    returns = equity[1:] / (equity[:-1] + 1e-8) - 1.0
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.0
    sortino = np.sqrt(252) * returns.mean() / (downside_std + 1e-8) if downside_std > 1e-12 else 0.0

    cumulative_return = equity[-1] / equity[0] - 1.0
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / (running_max + 1e-8)
    max_drawdown_abs = abs(drawdown.min())
    calmar = cumulative_return / (max_drawdown_abs + 1e-8)

    trade_count = int(sum(1 for a in action_history if a in (1, 2)))
    trade_density = trade_count / max(len(action_history), 1)

    return {
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "trade_count": trade_count,
        "trade_density": float(trade_density),
    }


def add_market_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    vol = labeled["volatility_10"].fillna(0.0)
    vol_high = float(vol.quantile(0.75))

    regimes = []
    for _, row in labeled.iterrows():
        ma5 = float(row.get("ma_5", row.get("Close", 0.0)))
        ma20 = float(row.get("ma_20", row.get("Close", 0.0)))
        ret1 = float(row.get("return_1", 0.0))
        v10 = float(row.get("volatility_10", 0.0))

        if v10 >= vol_high:
            regimes.append("high_volatility")
        elif ma5 > ma20 and ret1 >= 0:
            regimes.append("bull_trend")
        elif ma5 < ma20 and ret1 <= 0:
            regimes.append("bear_trend")
        else:
            regimes.append("range_bound")

    labeled["market_regime"] = regimes
    return labeled


def get_regime_thresholds(regime: str, base_threshold: float, strictness_multiplier: float):
    # Stricter thresholds in noisy markets, looser thresholds in trending markets.
    thresholds = {
        "bull_trend": {"buy": max(0.35, base_threshold - 0.10), "sell": min(0.80, base_threshold + 0.08)},
        "bear_trend": {"buy": min(0.80, base_threshold + 0.08), "sell": max(0.35, base_threshold - 0.10)},
        "range_bound": {"buy": base_threshold, "sell": base_threshold},
        "high_volatility": {
            "buy": min(0.90, base_threshold * strictness_multiplier),
            "sell": min(0.90, base_threshold * strictness_multiplier),
        },
    }
    return thresholds.get(regime, {"buy": base_threshold, "sell": base_threshold})


def infer_next_signal(model, df: pd.DataFrame, feature_cols: list, current_regime: str, base_threshold: float, strictness_multiplier: float):
    # Use the latest engineered features plus a neutral portfolio state as next-period input.
    feats = df[feature_cols].values.astype(np.float32)
    feat_mean = feats.mean(axis=0, keepdims=True)
    feat_std = feats.std(axis=0, keepdims=True) + 1e-8
    latest_feat = ((feats[-1:] - feat_mean) / feat_std).astype(np.float32)[0]

    agent_state = np.array([0.0, 1.0, 1.0], dtype=np.float32)  # position=0, balance=1x, net_worth=1x
    obs = np.concatenate([latest_feat, agent_state], axis=0).astype(np.float32).reshape(1, -1)

    action, _ = model.predict(obs, deterministic=True)

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    dist = model.policy.get_distribution(obs_tensor)
    probs = dist.distribution.probs.detach().cpu().numpy()[0]

    action_idx = int(np.asarray(action).reshape(-1)[0])
    action_map = {0: "Hold", 1: "Buy", 2: "Sell"}

    regime_thresholds = get_regime_thresholds(current_regime, base_threshold, strictness_multiplier)
    buy_prob = float(probs[1])
    sell_prob = float(probs[2])

    if buy_prob >= regime_thresholds["buy"] and buy_prob > sell_prob:
        gated_action = 1
    elif sell_prob >= regime_thresholds["sell"] and sell_prob > buy_prob:
        gated_action = 2
    else:
        gated_action = 0

    return {
        "action": gated_action,
        "label": action_map.get(gated_action, "Hold"),
        "raw_action": action_idx,
        "raw_label": action_map.get(action_idx, "Hold"),
        "confidence": float(np.max(probs)),
        "probs": probs,
        "thresholds": regime_thresholds,
    }


def run_walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: list,
    initial_balance: float,
    trade_fee: float,
    train_window: int,
    test_window: int,
    max_folds: int,
    timesteps_per_fold: int,
    slippage_bps: float,
    spread_bps: float,
    maker_fee: float,
    taker_fee: float,
    min_trade_pct: float,
    min_notional: float,
    min_qty: float,
    qty_step: float,
    price_step: float,
    position_step: float,
    slippage_vol_multiplier: float,
    max_drawdown_limit: float,
    daily_loss_limit: float,
    volatility_target: float,
):
    rows = []
    fold = 0
    start = 0

    while fold < max_folds and (start + train_window + test_window) <= len(df):
        fold += 1
        train_df = df.iloc[start : start + train_window].reset_index(drop=True)
        test_df = df.iloc[start + train_window : start + train_window + test_window].reset_index(drop=True)

        def make_train_env(local_train_df=train_df):
            return BitcoinTradingEnv(
                df=local_train_df,
                feature_cols=feature_cols,
                initial_balance=initial_balance,
                trade_fee=trade_fee,
                slippage_bps=slippage_bps,
                spread_bps=spread_bps,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                min_trade_pct=min_trade_pct,
                min_notional=min_notional,
                min_qty=min_qty,
                qty_step=qty_step,
                price_step=price_step,
                position_step=position_step,
                slippage_vol_multiplier=slippage_vol_multiplier,
                max_drawdown_limit=max_drawdown_limit,
                daily_loss_limit=daily_loss_limit,
                volatility_target=volatility_target,
            )

        train_env = DummyVecEnv([make_train_env])
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=5e-4,
            n_steps=1024,
            batch_size=32,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            verbose=0,
            device="auto",
            seed=100 + fold,
        )
        model.learn(total_timesteps=int(timesteps_per_fold))

        test_env = BitcoinTradingEnv(
            df=test_df,
            feature_cols=feature_cols,
            initial_balance=initial_balance,
            trade_fee=trade_fee,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            min_trade_pct=min_trade_pct,
            min_notional=min_notional,
            min_qty=min_qty,
            qty_step=qty_step,
            price_step=price_step,
            position_step=position_step,
            slippage_vol_multiplier=slippage_vol_multiplier,
            max_drawdown_limit=max_drawdown_limit,
            daily_loss_limit=daily_loss_limit,
            volatility_target=volatility_target,
        )

        metrics, equity_curve, action_history, _ = evaluate_agent(model, test_env)
        adv = compute_advanced_metrics(equity_curve, action_history)

        rows.append(
            {
                "Fold": fold,
                "Start": int(start),
                "End": int(start + train_window + test_window),
                "CumulativeReturn": float(metrics["cumulative_return"]),
                "Sharpe": float(metrics["sharpe_ratio"]),
                "MaxDrawdown": float(metrics["max_drawdown"]),
                "Sortino": float(adv["sortino_ratio"]),
                "Calmar": float(adv["calmar_ratio"]),
            }
        )

        start += test_window

    return pd.DataFrame(rows)


def run_cost_stress_test(
    model,
    test_df: pd.DataFrame,
    feature_cols: list,
    initial_balance: float,
    trade_fee: float,
    slippage_bps: float,
    spread_bps: float,
    maker_fee: float,
    taker_fee: float,
    min_trade_pct: float,
    min_notional: float,
    min_qty: float,
    qty_step: float,
    price_step: float,
    position_step: float,
    slippage_vol_multiplier: float,
    max_drawdown_limit: float,
    daily_loss_limit: float,
    volatility_target: float,
):
    scenarios = [
        ("Base", trade_fee, slippage_bps, spread_bps),
        ("Fee x2", trade_fee * 2.0, slippage_bps, spread_bps),
        ("Slippage x2", trade_fee, slippage_bps * 2.0, spread_bps),
        ("Spread x2", trade_fee, slippage_bps, spread_bps * 2.0),
    ]

    rows = []
    for name, fee, slip, spr in scenarios:
        env = BitcoinTradingEnv(
            df=test_df,
            feature_cols=feature_cols,
            initial_balance=initial_balance,
            trade_fee=float(fee),
            slippage_bps=float(slip),
            spread_bps=float(spr),
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            min_trade_pct=min_trade_pct,
            min_notional=min_notional,
            min_qty=min_qty,
            qty_step=qty_step,
            price_step=price_step,
            position_step=position_step,
            slippage_vol_multiplier=slippage_vol_multiplier,
            max_drawdown_limit=max_drawdown_limit,
            daily_loss_limit=daily_loss_limit,
            volatility_target=volatility_target,
        )
        metrics, equity_curve, _, _ = evaluate_agent(model, env)
        rows.append(
            {
                "Scenario": name,
                "FinalNetWorth": float(equity_curve[-1]),
                "CumulativeReturn": float(metrics["cumulative_return"]),
                "Sharpe": float(metrics["sharpe_ratio"]),
                "MaxDrawdown": float(metrics["max_drawdown"]),
            }
        )

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 執行
# ──────────────────────────────────────────────
if run_btn:
    try:
        # 1) 資料
        df_raw = load_or_download_data()

        with st.spinner("計算技術指標..."):
            df = add_technical_indicators_cached(df_raw)
        df = add_market_regime_labels(df)

        if performance_mode == "快速模式" and len(df) > int(fast_max_bars):
            df = df.iloc[-int(fast_max_bars):].reset_index(drop=True)
            st.info(f"快速模式已啟用：僅使用最近 {len(df)} 筆資料加速訓練。")

        feature_cols = build_feature_columns()

        # 2) 切分
        split_idx = int(len(df) * train_split)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df  = df.iloc[split_idx:].reset_index(drop=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("總資料筆數", len(df))
        col2.metric("訓練集", len(train_df))
        col3.metric("測試集", len(test_df))

        # 3) 訓練環境
        def make_train_env():
            return BitcoinTradingEnv(
                df=train_df,
                feature_cols=feature_cols,
                initial_balance=float(initial_balance),
                trade_fee=trade_fee,
                slippage_bps=slippage_bps,
                spread_bps=spread_bps,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                min_trade_pct=min_trade_pct,
                min_notional=min_notional,
                min_qty=min_qty,
                qty_step=qty_step,
                price_step=price_step,
                position_step=float(position_step),
                slippage_vol_multiplier=slippage_vol_multiplier,
                max_drawdown_limit=max_drawdown_limit,
                daily_loss_limit=daily_loss_limit,
                volatility_target=volatility_target,
            )

        train_env = DummyVecEnv([make_train_env])

        effective_timesteps = int(total_timesteps)
        if performance_mode == "快速模式":
            effective_timesteps = min(effective_timesteps, 50_000)

        # 4) 訓練 PPO
        model_loaded = False
        model_path_zip = f"{MODEL_PATH}.zip"
        if use_saved_model and os.path.exists(model_path_zip):
            with st.spinner("載入既有模型中..."):
                model = PPO.load(MODEL_PATH, env=train_env, device="auto")
                model_loaded = True
                st.info("已載入既有模型，跳過重新訓練。")

        if not model_loaded:
            with st.spinner(f"訓練 PPO 模型中（{effective_timesteps:,} 步）..."):
                policy_kwargs = dict(
                    net_arch=[256, 256, 128],
                )

                if performance_mode == "快速模式":
                    n_steps = 512
                    batch_size = 64
                    n_epochs = 8
                else:
                    n_steps = 1024
                    batch_size = 32
                    n_epochs = 15

                model = PPO(
                    policy="MlpPolicy",
                    env=train_env,
                    learning_rate=5e-4,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.02,
                    verbose=0,
                    device="auto",
                    policy_kwargs=policy_kwargs,
                    seed=42,
                )
                model.learn(total_timesteps=effective_timesteps)
                model.save(MODEL_PATH)

        st.success(f"✅ 模型訓練完成，已儲存為 `{MODEL_PATH}.zip`")

        # 5) 測試評估
        test_env = BitcoinTradingEnv(
            df=test_df,
            feature_cols=feature_cols,
            initial_balance=float(initial_balance),
            trade_fee=trade_fee,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            min_trade_pct=min_trade_pct,
            min_notional=min_notional,
            min_qty=min_qty,
            qty_step=qty_step,
            price_step=price_step,
            position_step=float(position_step),
            slippage_vol_multiplier=slippage_vol_multiplier,
            max_drawdown_limit=max_drawdown_limit,
            daily_loss_limit=daily_loss_limit,
            volatility_target=volatility_target,
        )
        metrics, equity_curve, action_history, trade_log = evaluate_agent(model, test_env)

        stress_df = None
        if run_stress_test:
            stress_df = run_cost_stress_test(
                model=model,
                test_df=test_df,
                feature_cols=feature_cols,
                initial_balance=float(initial_balance),
                trade_fee=trade_fee,
                slippage_bps=slippage_bps,
                spread_bps=spread_bps,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                min_trade_pct=min_trade_pct,
                min_notional=min_notional,
                min_qty=min_qty,
                qty_step=qty_step,
                price_step=price_step,
                position_step=float(position_step),
                slippage_vol_multiplier=slippage_vol_multiplier,
                max_drawdown_limit=max_drawdown_limit,
                daily_loss_limit=daily_loss_limit,
                volatility_target=volatility_target,
            )

        # Buy & Hold baseline
        test_prices = test_df["Close"].values
        buy_hold_curve = float(initial_balance) * (test_prices / test_prices[0])
        time_axis, use_datetime = get_time_axis(test_df, len(equity_curve))

        # 6) 指標顯示
        st.subheader("📊 測試集績效指標")
        m1, m2, m3, m4 = st.columns(4)
        cr  = metrics["cumulative_return"]
        bh_cr = (buy_hold_curve[len(equity_curve) - 1] / float(initial_balance)) - 1.0
        m1.metric("累積報酬率（RL）",  f"{cr * 100:.2f} %", delta=f"{(cr - bh_cr) * 100:.2f} % vs B&H")
        m2.metric("Sharpe Ratio",       f"{metrics['sharpe_ratio']:.3f}")
        m3.metric("最大回撤",            f"{metrics['max_drawdown'] * 100:.2f} %")
        m4.metric("最終資產（USD）",     f"{equity_curve[-1]:,.2f}")

        # 6-0) 目前市場 Regime
        current_regime = str(df["market_regime"].iloc[-1])
        regime_name_map = {
            "bull_trend": "多頭趨勢",
            "bear_trend": "空頭趨勢",
            "range_bound": "盤整震盪",
            "high_volatility": "高波動",
        }
        regime_cn = regime_name_map.get(current_regime, current_regime)
        st.subheader("🌦️ 目前市場 Regime")
        r1, r2, r3 = st.columns(3)
        r1.metric("Regime", regime_cn)
        regime_thresholds = get_regime_thresholds(current_regime, base_threshold, strictness_multiplier)
        r2.metric("Buy 閾值", f"{regime_thresholds['buy'] * 100:.1f} %")
        r3.metric("Sell 閾值", f"{regime_thresholds['sell'] * 100:.1f} %")

        # 6-1) 下一根 K 棒訊號（1d 時可視為明日訊號）
        next_signal = infer_next_signal(
            model=model,
            df=df,
            feature_cols=feature_cols,
            current_regime=current_regime,
            base_threshold=base_threshold,
            strictness_multiplier=strictness_multiplier,
        )
        st.subheader("🔮 下一根 K 棒建議訊號")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("建議動作", next_signal["label"])
        s2.metric("信心分數", f"{next_signal['confidence'] * 100:.1f} %")
        s3.metric("Buy 機率", f"{next_signal['probs'][1] * 100:.1f} %")
        s4.metric("Sell 機率", f"{next_signal['probs'][2] * 100:.1f} %")
        st.caption(
            f"原始策略動作: {next_signal['raw_label']}｜"
            f"Regime 閾值 Buy>={next_signal['thresholds']['buy'] * 100:.1f}% / "
            f"Sell>={next_signal['thresholds']['sell'] * 100:.1f}%"
        )

        # 6-2) 進階風險指標
        adv = compute_advanced_metrics(equity_curve, action_history)
        st.subheader("🧪 進階風險指標")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Sortino Ratio", f"{adv['sortino_ratio']:.3f}")
        a2.metric("Calmar Ratio", f"{adv['calmar_ratio']:.3f}")
        a3.metric("交易次數", f"{adv['trade_count']}")
        a4.metric("交易密度", f"{adv['trade_density'] * 100:.1f} %")

        if stress_df is not None:
            st.subheader("🧱 成本壓力測試")
            st.dataframe(stress_df, use_container_width=True)
            stress_chart = stress_df.set_index("Scenario")[["FinalNetWorth"]]
            st.bar_chart(stress_chart)

        st.subheader("🧾 交易執行日誌")
        if len(trade_log) > 0:
            trade_df = pd.DataFrame(trade_log)
            st.dataframe(trade_df.tail(100), use_container_width=True)
            exec_mix = trade_df["execution"].value_counts().rename_axis("Execution").to_frame("Count")
            st.bar_chart(exec_mix)
        else:
            st.info("本次測試無成交紀錄（可能被風險或門檻過濾）。")

        if yf_interval == "1d":
            st.info("此訊號對應下一根日線（可視為明日建議）。")
        else:
            st.info("此訊號對應下一根 K 棒（你目前使用的是 1h 週期）。")

        # 7) 圖表
        st.subheader("📈 資產曲線")
        st.pyplot(plot_equity_curve(equity_curve, buy_hold_curve, time_axis, use_datetime))

        st.subheader("🔔 交易訊號")
        st.pyplot(plot_price_with_signals(test_df, action_history, time_axis, use_datetime))

        st.subheader("🗺️ Regime 背景視圖")
        st.pyplot(plot_price_with_regime_overlay(test_df, action_history, time_axis, use_datetime))

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🎯 動作分佈")
            st.pyplot(plot_action_distribution(action_history))

        with col_b:
            st.subheader("📉 資產曲線（資料）")
            eq_df = pd.DataFrame({
                "RL Agent": equity_curve,
                "Buy & Hold": buy_hold_curve[: len(equity_curve)],
            })
            if use_datetime:
                eq_df["Datetime"] = time_axis
                eq_df = eq_df.set_index("Datetime")
            st.line_chart(eq_df)

        effective_walk_forward = enable_walk_forward and performance_mode == "完整模式"
        if enable_walk_forward and performance_mode == "快速模式":
            st.info("快速模式下已自動略過 Walk-forward，以縮短等待時間。")

        if effective_walk_forward:
            with st.spinner("執行 Walk-forward 滾動回測中..."):
                wf_df = run_walk_forward_backtest(
                    df=df,
                    feature_cols=feature_cols,
                    initial_balance=float(initial_balance),
                    trade_fee=trade_fee,
                    train_window=int(wf_train_window),
                    test_window=int(wf_test_window),
                    max_folds=int(wf_max_folds),
                    timesteps_per_fold=int(wf_timesteps),
                    slippage_bps=slippage_bps,
                    spread_bps=spread_bps,
                    maker_fee=maker_fee,
                    taker_fee=taker_fee,
                    min_trade_pct=min_trade_pct,
                    min_notional=min_notional,
                    min_qty=min_qty,
                    qty_step=qty_step,
                    price_step=price_step,
                    position_step=float(position_step),
                    slippage_vol_multiplier=slippage_vol_multiplier,
                    max_drawdown_limit=max_drawdown_limit,
                    daily_loss_limit=daily_loss_limit,
                    volatility_target=volatility_target,
                )

            if not wf_df.empty:
                st.subheader("🔁 Walk-forward 回測結果")
                st.dataframe(wf_df, use_container_width=True)
                wf_summary = pd.DataFrame(
                    {
                        "AvgReturn": [wf_df["CumulativeReturn"].mean()],
                        "AvgSharpe": [wf_df["Sharpe"].mean()],
                        "AvgMaxDrawdown": [wf_df["MaxDrawdown"].mean()],
                    }
                )
                st.dataframe(wf_summary, use_container_width=True)
                st.line_chart(wf_df.set_index("Fold")[["CumulativeReturn", "Sharpe"]])
            else:
                st.warning("Walk-forward 參數超出資料長度，請調小訓練/測試窗口或折數。")

        # 8) 訓練資料摘要
        with st.expander("📋 原始資料摘要"):
            st.dataframe(df_raw.tail(50), use_container_width=True)

    except Exception as e:
        st.error(f"❌ 發生錯誤：\n\n`{str(e)}`\n\n請檢查參數或資料後重新嘗試。")
        import traceback
        with st.expander("🔧 詳細錯誤訊息"):
            st.code(traceback.format_exc())

else:
    # 說明頁
    st.info("👈 在左側設定好參數後，按下「開始下載 & 訓練」即可執行完整流程。")

    st.subheader("📌 系統架構")
    st.markdown("""
| 模組 | 說明 |
|------|------|
| **資料來源** | yfinance 下載 BTC-USD 歷史 K 棒，或自行上傳 CSV |
| **特徵工程** | MA5/10/20、RSI(14)、MACD、波動率、成交量比等 13 個指標 |
| **交易環境** | 自定義 Gymnasium Env，離散動作：Hold / Buy / Sell |
| **RL 演算法** | Stable-Baselines3 **PPO**（MlpPolicy） |
| **評估指標** | 累積報酬率、Sharpe Ratio、最大回撤、對比 Buy & Hold 基準 |
""")

    st.subheader("🔄 PPO 演算法簡介")
    st.markdown("""
**Proximal Policy Optimization（PPO）** 是一種 on-policy 的 Actor-Critic 強化學習演算法：

1. **Actor（策略網路）** 輸出在當前狀態下選擇每個動作的機率
2. **Critic（價值網路）** 估計當前狀態的期望回報 $V(s)$
3. **Clipped Surrogate Objective** 限制策略更新幅度，避免過度偏離舊策略：

$$L^{CLIP}(\\theta) = \\mathbb{E}_t \\left[ \\min\\left( r_t(\\theta) \\hat{A}_t,\\ \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t \\right) \\right]$$

其中 $r_t(\\theta) = \\dfrac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{old}}(a_t|s_t)}$，$\\hat{A}_t$ 為 GAE 優勢估計。
""")
