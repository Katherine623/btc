# app_btc.py
# -*- coding: utf-8 -*-
"""
Bitcoin RL Trading (PPO) - Streamlit UI
Run with: streamlit run app_btc.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import streamlit as st

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from btc_rl_trading_ppo import (
    download_btc_data,
    load_data,
    add_technical_indicators,
    build_feature_columns,
    build_vanilla_feature_columns,
    build_attention_policy_kwargs,
    BitcoinTradingEnv,
    evaluate_agent,
    compute_metrics,
    evaluate_momentum_baseline,
    run_ppo_baseline,
    explain_actor_with_shap,
)

CJK_FONT_PROP = None
CJK_FONT_FAMILY = None


def configure_matplotlib_cjk_font():
    """Set a CJK-capable font fallback list to avoid garbled Chinese labels."""
    global CJK_FONT_PROP, CJK_FONT_FAMILY

    candidate_paths = [
        r"C:\Windows\Fonts\msjh.ttc",      # Microsoft JhengHei
        r"C:\Windows\Fonts\msjhbd.ttc",    # Microsoft JhengHei Bold
        r"C:\Windows\Fonts\msyh.ttc",      # Microsoft YaHei
        r"C:\Windows\Fonts\simhei.ttf",    # SimHei
    ]

    selected = None
    for font_path in candidate_paths:
        if os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
                CJK_FONT_PROP = fm.FontProperties(fname=font_path)
                selected = CJK_FONT_PROP.get_name()
                CJK_FONT_FAMILY = selected
                break
            except Exception:
                continue

    preferred_fonts = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "PingFang TC",
        "SimHei",
        "Arial Unicode MS",
    ]

    available = {font.name for font in fm.fontManager.ttflist}
    if not selected:
        selected = next((name for name in preferred_fonts if name in available), None)
        if selected:
            CJK_FONT_PROP = fm.FontProperties(family=selected)
            CJK_FONT_FAMILY = selected

    if selected:
        plt.rcParams["font.family"] = selected
    else:
        plt.rcParams["font.family"] = "sans-serif"
    existing = list(plt.rcParams.get("font.sans-serif", []))
    if selected:
        plt.rcParams["font.sans-serif"] = [selected] + [name for name in existing if name != selected]
    else:
        # Keep defaults but still avoid minus sign glyph issues.
        plt.rcParams["font.sans-serif"] = existing

    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_cjk_font()


def cjk_text_kwargs(**kwargs):
    if CJK_FONT_PROP is not None:
        kwargs["fontproperties"] = CJK_FONT_PROP
    elif CJK_FONT_FAMILY:
        kwargs["fontfamily"] = CJK_FONT_FAMILY
    return kwargs

# ──────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bitcoin RL Trading Model",
    page_icon="₿",
    layout="wide",
)

st.title("₿ Bitcoin Reinforcement Learning Trading Model (PPO)")
st.markdown(
    "Train a BTC trading agent with **Proximal Policy Optimization (PPO)**, "
    "and learn buy / sell / hold strategies automatically."
)

# ──────────────────────────────────────────────
# Sidebar: data & hyperparameters
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Data Source")
    data_source = st.radio(
        "Choose a data source",
        ["Download from yfinance", "Upload CSV file"],
        index=0,
    )

    yf_period = st.selectbox(
        "Download period",
        ["1y", "2y", "5y", "max"],
        index=1,
    )
    yf_interval = st.selectbox(
        "Bar interval",
        ["1d", "1h"],
        index=0,
    )

    uploaded_file = None
    if data_source == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload CSV (must include Open/High/Low/Close/Volume)", type=["csv"])

    st.divider()

    st.subheader("Training Hyperparameters")
    initial_balance = st.number_input("Initial balance (USD)", value=10000, min_value=100, step=500)
    trade_fee = st.slider("Trading fee", min_value=0.0, max_value=0.01, value=0.001, step=0.0005, format="%.4f")
    total_timesteps = st.select_slider(
        "Total training timesteps",
        options=[50_000, 100_000, 200_000, 300_000, 500_000],
        value=200_000,
    )
    train_split = st.slider("Training split ratio", min_value=0.5, max_value=0.9, value=0.8, step=0.05)

    st.divider()
    st.subheader("⚡ Execution Speed")
    performance_mode = st.radio(
        "Run mode",
        ["Fast mode", "Full mode"],
        index=0,
        horizontal=True,
    )
    use_saved_model = st.checkbox("Prefer loading an existing model if available", value=True)
    run_stress_test = st.checkbox("Enable cost stress test", value=(performance_mode == "Full mode"))
    run_benchmark_suite = st.checkbox("Enable benchmark suite (B&H / Momentum / Vanilla PPO)", value=(performance_mode == "Full mode"))
    run_shap_analysis = st.checkbox("Enable SHAP / feature contribution analysis", value=False)
    if performance_mode == "Fast mode":
        fast_max_bars = st.number_input("Max bars in fast mode", min_value=300, max_value=3000, value=900, step=100)
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
        "action_threshold": 0.10,
        "lambda_downside": 0.12,
        "eta_trade_penalty": 0.004,
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
        st.session_state.strategy_preset = "Balanced"

    ui_mode = st.radio(
        "Parameter mode",
        ["Auto mode", "Advanced mode"],
        index=0,
        horizontal=True,
    )

    st.subheader("🎛️ Strategy Style")
    st.caption("One-click presets: Conservative / Balanced / Aggressive")
    p1, p2, p3 = st.columns(3)
    if p1.button("Conservative", use_container_width=True):
        st.session_state.base_threshold = 0.62
        st.session_state.strictness_multiplier = 1.30
        st.session_state.min_trade_pct = 0.03
        st.session_state.position_step = 0.20
        st.session_state.enable_walk_forward = False
        st.session_state.strategy_preset = "Conservative"
    if p2.button("Balanced", use_container_width=True):
        st.session_state.base_threshold = 0.55
        st.session_state.strictness_multiplier = 1.15
        st.session_state.min_trade_pct = 0.02
        st.session_state.position_step = 0.25
        st.session_state.enable_walk_forward = False
        st.session_state.strategy_preset = "Balanced"
    if p3.button("Aggressive", use_container_width=True):
        st.session_state.base_threshold = 0.48
        st.session_state.strictness_multiplier = 1.05
        st.session_state.min_trade_pct = 0.01
        st.session_state.position_step = 0.33
        st.session_state.enable_walk_forward = False
        st.session_state.strategy_preset = "Aggressive"

    if ui_mode == "Auto mode":
        st.info(
            f"Currently using the {st.session_state.strategy_preset} preset: "
            f"Regime threshold {st.session_state.base_threshold:.2f} / "
            f"high-volatility multiplier {st.session_state.strictness_multiplier:.2f} / "
            f"rebalance step {st.session_state.position_step:.2f}"
        )
    else:
        st.divider()
        st.subheader("🏦 Realistic Market Settings")
        st.session_state.slippage_bps = st.slider("Base slippage (bps)", min_value=0.0, max_value=30.0, value=float(st.session_state.slippage_bps), step=1.0)
        st.session_state.spread_bps = st.slider("Bid-ask spread (bps)", min_value=0.0, max_value=20.0, value=float(st.session_state.spread_bps), step=1.0)
        st.session_state.maker_fee = st.slider("Maker fee", min_value=0.0, max_value=0.0020, value=float(st.session_state.maker_fee), step=0.0001, format="%.4f")
        st.session_state.taker_fee = st.slider("Taker fee", min_value=0.0, max_value=0.0030, value=float(st.session_state.taker_fee), step=0.0001, format="%.4f")
        st.session_state.min_trade_pct = st.slider("Minimum trade size ratio", min_value=0.0, max_value=0.10, value=float(st.session_state.min_trade_pct), step=0.005)
        st.session_state.min_notional = st.number_input("Minimum notional (USD)", min_value=1.0, value=float(st.session_state.min_notional), step=1.0)
        st.session_state.min_qty = st.number_input("Minimum order size (BTC)", min_value=0.00001, value=float(st.session_state.min_qty), step=0.00001, format="%.5f")
        st.session_state.qty_step = st.number_input("Quantity step", min_value=0.00001, value=float(st.session_state.qty_step), step=0.00001, format="%.5f")
        st.session_state.price_step = st.number_input("Price step", min_value=0.01, value=float(st.session_state.price_step), step=0.01)
        st.session_state.position_step = st.select_slider("Rebalance step", options=[0.10, 0.20, 0.25, 0.33, 0.50], value=float(st.session_state.position_step))
        st.session_state.slippage_vol_multiplier = st.slider("High-volatility slippage multiplier", min_value=0.0, max_value=3.0, value=float(st.session_state.slippage_vol_multiplier), step=0.1)
        st.session_state.volatility_target = st.slider("Volatility target (risk scaling)", min_value=0.005, max_value=0.05, value=float(st.session_state.volatility_target), step=0.001, format="%.3f")
        st.session_state.action_threshold = st.slider("Action smoothing threshold |ΔA|", min_value=0.01, max_value=0.30, value=float(st.session_state.action_threshold), step=0.01)
        st.session_state.lambda_downside = st.slider("Downside risk weight λ", min_value=0.01, max_value=0.30, value=float(st.session_state.lambda_downside), step=0.01)
        st.session_state.eta_trade_penalty = st.slider("Rebalance penalty weight η", min_value=0.0005, max_value=0.02, value=float(st.session_state.eta_trade_penalty), step=0.0005, format="%.4f")

        st.divider()
        st.subheader("🛑 Risk Engine")
        st.session_state.max_drawdown_limit = st.slider("Max drawdown stop line", min_value=0.10, max_value=0.60, value=float(st.session_state.max_drawdown_limit), step=0.01)
        st.session_state.daily_loss_limit = st.slider("Per-run loss stop line", min_value=0.02, max_value=0.30, value=float(st.session_state.daily_loss_limit), step=0.01)

        st.divider()
        st.subheader("🧪 Walk-forward Backtest")
        st.session_state.enable_walk_forward = st.checkbox("Enable walk-forward rolling backtest", value=bool(st.session_state.enable_walk_forward))
        if st.session_state.enable_walk_forward:
            st.session_state.wf_train_window = int(st.number_input("Train window per fold (bars)", min_value=120, value=int(st.session_state.wf_train_window), step=60))
            st.session_state.wf_test_window = int(st.number_input("Test window per fold (bars)", min_value=48, value=int(st.session_state.wf_test_window), step=24))
            st.session_state.wf_max_folds = int(st.number_input("Max folds", min_value=2, max_value=12, value=int(st.session_state.wf_max_folds), step=1))
            st.session_state.wf_timesteps = int(
                st.select_slider(
                    "Timesteps per fold",
                    options=[10_000, 20_000, 30_000, 50_000, 80_000],
                    value=int(st.session_state.wf_timesteps),
                )
            )

        st.divider()
        st.subheader("🧠 Regime Threshold Control")
        st.session_state.base_threshold = st.slider(
            "Base confidence threshold",
            min_value=0.45,
            max_value=0.75,
            value=float(st.session_state.base_threshold),
            step=0.01,
        )
        st.session_state.strictness_multiplier = st.slider(
            "High-volatility strictness multiplier",
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
    action_threshold = float(st.session_state.action_threshold)
    lambda_downside = float(st.session_state.lambda_downside)
    eta_trade_penalty = float(st.session_state.eta_trade_penalty)

    enable_walk_forward = bool(st.session_state.enable_walk_forward)
    wf_train_window = int(st.session_state.wf_train_window)
    wf_test_window = int(st.session_state.wf_test_window)
    wf_max_folds = int(st.session_state.wf_max_folds)
    wf_timesteps = int(st.session_state.wf_timesteps)
    base_threshold = float(st.session_state.base_threshold)
    strictness_multiplier = float(st.session_state.strictness_multiplier)

    st.divider()
    run_btn = st.button("🚀 Start Download & Train", use_container_width=True)

# ──────────────────────────────────────────────
# Main flow
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
    if data_source == "Upload CSV file" and uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ CSV read failed: {str(e)}")
            st.stop()
    elif data_source == "Download from yfinance":
        try:
            with st.spinner("Downloading BTC data from yfinance... (this may take 10-30 seconds)"):
                df_raw = fetch_yfinance_cached(yf_interval, yf_period, CSV_PATH)
        except Exception as e:
            st.error(
                f"❌ yfinance download failed.\n\n"
                f"**Reason:** {str(e)}\n\n"
                f"**Suggestions:**\n"
                f"1. Wait 1-2 minutes and try again (API rate limiting)\n"
                f"2. Switch to the 'Upload CSV file' option\n"
                f"3. Use a shorter period (for example 1y instead of max)"
            )
            st.stop()
    elif os.path.exists(CSV_PATH):
        df_raw = pd.read_csv(CSV_PATH)
        st.info(f"✓ Loaded existing data file: {CSV_PATH} ({len(df_raw)} rows)")
    else:
        st.error("❌ Please choose a data source or upload a CSV before running.")
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
    labels = ["Sell (-1)", "Hold (0)", "Buy (+1)"]
    counts = [action_history.count(-1), action_history.count(0), action_history.count(1)]
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
    sell_steps = [i for i, a in enumerate(action_history) if a == -1]

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
    sell_steps = [i for i, a in enumerate(action_history) if a == -1]

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


def _calc_hold_stats(action_history):
    # Non-zero action implies an executed trade. Hold length is bars between trades.
    trade_steps = [i for i, a in enumerate(action_history) if a != 0]
    if len(trade_steps) < 2:
        return 0, 0
    gaps = np.diff(trade_steps)
    return int(np.round(np.mean(gaps))), int(np.max(gaps))


def plot_performance_dashboard(equity_curve, buy_hold_curve, time_axis, use_datetime, action_history):
    eq = np.asarray(equity_curve, dtype=np.float64)
    bh = np.asarray(buy_hold_curve[: len(eq)], dtype=np.float64)

    eq_ret = eq[1:] / (eq[:-1] + 1e-8) - 1.0 if len(eq) > 2 else np.array([0.0])
    bh_ret = bh[1:] / (bh[:-1] + 1e-8) - 1.0 if len(bh) > 2 else np.array([0.0])

    ann_factor = np.sqrt(252)
    annual_return = (eq[-1] / (eq[0] + 1e-8) - 1.0) * 100.0
    alpha = (eq_ret.mean() - bh_ret.mean()) * 252 * 100.0
    beta = float(np.cov(eq_ret, bh_ret)[0, 1] / (np.var(bh_ret) + 1e-8)) if len(eq_ret) > 3 else 0.0
    avg_hold, max_hold = _calc_hold_stats(action_history)

    if use_datetime:
        dt = pd.to_datetime(time_axis[: len(eq)], errors="coerce")
        years = pd.Series(dt).dt.year.ffill().fillna(0).astype(int)
    else:
        # Fallback pseudo-years for non-datetime index.
        years = pd.Series(np.floor(np.linspace(2020, 2020 + len(eq) / 250, len(eq))).astype(int))

    years_unique = [y for y in sorted(years.unique()) if y > 0]
    yearly_returns = []
    for y in years_unique:
        idx = np.where(years.values == y)[0]
        if len(idx) < 2:
            yearly_returns.append(0.0)
        else:
            r = eq[idx[-1]] / (eq[idx[0]] + 1e-8) - 1.0
            yearly_returns.append(float(r * 100.0))

    fig = plt.figure(figsize=(12, 8), facecolor="#111325")
    gs = fig.add_gridspec(12, 24)

    # Top KPI cards
    ax_cards = fig.add_subplot(gs[0:3, :])
    ax_cards.set_facecolor("#111325")
    ax_cards.axis("off")

    card_labels = ["Annual Return", "Alpha", "Beta", "Avg Hold", "Max Hold"]
    card_values = [
        f"{annual_return:+.1f}%",
        f"{alpha:+.1f}%",
        f"{beta:.2f}",
        f"{avg_hold} bars",
        f"{max_hold} bars",
    ]
    card_flags = [annual_return > 0, alpha > 0, beta < 1.0, avg_hold >= 5, max_hold >= avg_hold]

    n = len(card_labels)
    for i in range(n):
        x0 = 0.02 + i * (0.96 / n)
        w = 0.96 / n - 0.01
        rect = patches.FancyBboxPatch(
            (x0, 0.12),
            w,
            0.76,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.8,
            edgecolor="#3A3E60",
            facecolor="#161934",
            transform=ax_cards.transAxes,
        )
        ax_cards.add_patch(rect)

        icon = "✓" if card_flags[i] else "✕"
        icon_color = "#63F5DD" if card_flags[i] else "#FF6B6B"
        ax_cards.text(x0 + 0.03, 0.72, icon, color=icon_color, fontsize=12, weight="bold", transform=ax_cards.transAxes, **cjk_text_kwargs())
        ax_cards.text(x0 + 0.06, 0.70, card_labels[i], color="#D8DCEC", fontsize=12, transform=ax_cards.transAxes, **cjk_text_kwargs())
        ax_cards.text(x0 + 0.02, 0.36, card_values[i], color="#EEF1FF", fontsize=22, weight="bold", transform=ax_cards.transAxes, **cjk_text_kwargs())

    # Main performance chart
    ax_main = fig.add_subplot(gs[3:9, :])
    ax_main.set_facecolor("#161934")
    ax_main.plot(eq, color="#7B6DFF", linewidth=2.2, label="SMC-PPO")
    ax_main.plot(bh, color="#A7ACBD", linewidth=2.0, alpha=0.9, label="Buy & Hold")

    ax_main.set_title("Historical Performance", loc="left", color="#ECEFFF", fontsize=24, fontweight="bold", pad=12, **cjk_text_kwargs())
    ax_main.tick_params(colors="#B7BDCF", labelsize=10)
    for spine in ax_main.spines.values():
        spine.set_color("#3A3E60")
    ax_main.grid(True, color="#2A2F4F", alpha=0.4)

    eq_pct = (eq / (eq[0] + 1e-8) - 1.0) * 100
    bh_pct = (bh / (bh[0] + 1e-8) - 1.0) * 100
    ax_right = ax_main.twinx()
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_yticks(np.linspace(ax_main.get_ylim()[0], ax_main.get_ylim()[1], 5))
    ax_right.set_yticklabels([f"{v:.2f}%" for v in np.linspace(eq_pct.min(), eq_pct.max(), 5)], color="#9FA5BB")
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    ax_main.annotate(
        f"{eq_pct[-1]:.2f}%",
        xy=(len(eq) - 1, eq[-1]),
        xytext=(-40, 0),
        textcoords="offset points",
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc="#7B6DFF", ec="none", alpha=0.95),
    )
    ax_main.annotate(
        f"{bh_pct[-1]:.2f}%",
        xy=(len(bh) - 1, bh[-1]),
        xytext=(-40, -10),
        textcoords="offset points",
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc="#8A8E9C", ec="none", alpha=0.95),
    )

    # Annual return strip
    ax_strip = fig.add_subplot(gs[9:12, :])
    ax_strip.set_facecolor("#161934")
    ax_strip.set_xlim(0, max(1, len(years_unique)))
    ax_strip.set_ylim(0, 1)
    ax_strip.axis("off")

    for i, (year, ret) in enumerate(zip(years_unique, yearly_returns)):
        color = "#A83A72" if ret >= 0 else "#4B4FA4"
        rect = patches.Rectangle((i + 0.02, 0.20), 0.96, 0.26, facecolor=color, edgecolor="none", alpha=0.95)
        ax_strip.add_patch(rect)
        ax_strip.text(i + 0.5, 0.58, f"{year}", ha="center", va="center", color="#D8DCEC", fontsize=10, **cjk_text_kwargs())
        ax_strip.text(i + 0.5, 0.33, f"{ret:.1f}%", ha="center", va="center", color="#F3F5FF", fontsize=12, weight="bold", **cjk_text_kwargs())

    fig.tight_layout(pad=1.1)
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

    trade_count = int(sum(1 for a in action_history if a != 0))
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
    # Continuous-action inference with regime gating.
    feats = df[feature_cols].values.astype(np.float32)
    feat_mean = feats.mean(axis=0, keepdims=True)
    feat_std = feats.std(axis=0, keepdims=True) + 1e-8
    latest_feat = ((feats[-1:] - feat_mean) / feat_std).astype(np.float32)[0]

    agent_state = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    obs = np.concatenate([latest_feat, agent_state], axis=0).astype(np.float32).reshape(1, -1)

    action, _ = model.predict(obs, deterministic=True)
    raw_action = float(np.asarray(action).reshape(-1)[0])

    regime_thresholds = get_regime_thresholds(current_regime, base_threshold, strictness_multiplier)
    buy_score = max(0.0, raw_action)
    sell_score = max(0.0, -raw_action)

    if buy_score >= regime_thresholds["buy"] and buy_score > sell_score:
        gated_action = 1
    elif sell_score >= regime_thresholds["sell"] and sell_score > buy_score:
        gated_action = -1
    else:
        gated_action = 0

    label_map = {1: "Buy", -1: "Sell", 0: "Hold"}

    return {
        "action": gated_action,
        "label": label_map.get(gated_action, "Hold"),
        "raw_action": raw_action,
        "raw_label": "Buy" if raw_action > 0.05 else ("Sell" if raw_action < -0.05 else "Hold"),
        "confidence": float(abs(raw_action)),
        "scores": np.array([sell_score, buy_score], dtype=np.float32),
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
    action_threshold: float,
    lambda_downside: float,
    eta_trade_penalty: float,
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
                action_threshold=action_threshold,
                lambda_downside=lambda_downside,
                eta_trade_penalty=eta_trade_penalty,
            )

        train_env = DummyVecEnv([make_train_env])
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=1e-4,
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
            policy_kwargs=build_attention_policy_kwargs(market_feature_dim=len(feature_cols), agent_state_dim=4),
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
            action_threshold=action_threshold,
            lambda_downside=lambda_downside,
            eta_trade_penalty=eta_trade_penalty,
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
    action_threshold: float,
    lambda_downside: float,
    eta_trade_penalty: float,
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
            action_threshold=action_threshold,
            lambda_downside=lambda_downside,
            eta_trade_penalty=eta_trade_penalty,
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
# Execution
# ──────────────────────────────────────────────
if run_btn:
    try:
        # 1) Data
        df_raw = load_or_download_data()

        with st.spinner("Computing technical indicators..."):
            df = add_technical_indicators_cached(df_raw)
        df = add_market_regime_labels(df)

        if performance_mode == "Fast mode" and len(df) > int(fast_max_bars):
            df = df.iloc[-int(fast_max_bars):].reset_index(drop=True)
            st.info(f"Fast mode enabled: only the most recent {len(df)} rows are used to speed up training.")

        feature_cols = build_feature_columns()
        feature_cols = [c for c in feature_cols if c in df.columns]
        vanilla_feature_cols = [c for c in build_vanilla_feature_columns() if c in df.columns]

        # 2) Split data
        split_idx = int(len(df) * train_split)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df  = df.iloc[split_idx:].reset_index(drop=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total rows", len(df))
        col2.metric("Train set", len(train_df))
        col3.metric("Test set", len(test_df))

        # 3) Training environment
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
                action_threshold=action_threshold,
                lambda_downside=lambda_downside,
                eta_trade_penalty=eta_trade_penalty,
            )

        train_env = DummyVecEnv([make_train_env])

        effective_timesteps = int(total_timesteps)
        if performance_mode == "Fast mode":
            effective_timesteps = min(effective_timesteps, 50_000)

        # 4) Train PPO
        model_loaded = False
        model_path_zip = f"{MODEL_PATH}.zip"
        if use_saved_model and os.path.exists(model_path_zip):
            with st.spinner("Loading existing model..."):
                try:
                    model = PPO.load(MODEL_PATH, env=train_env, device="auto")
                    model_loaded = True
                    st.info("Existing model loaded; retraining skipped.")
                except Exception as load_err:
                    st.warning(
                        "The existing model is incompatible with the current feature dimensions / observation space. "
                        "Switching to retraining automatically.\n\n"
                        f"Reason: {str(load_err)}"
                    )
                    model_loaded = False

        if not model_loaded:
            with st.spinner(f"Training PPO model ({effective_timesteps:,} steps)..."):
                policy_kwargs = build_attention_policy_kwargs(market_feature_dim=len(feature_cols), agent_state_dim=4)

                if performance_mode == "Fast mode":
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
                    learning_rate=1e-4,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    verbose=0,
                    device="auto",
                    policy_kwargs=policy_kwargs,
                    seed=42,
                )
                model.learn(total_timesteps=effective_timesteps)
                model.save(MODEL_PATH)

        st.success(f"✅ Model training completed and saved as `{MODEL_PATH}.zip`")

        # 5) Test evaluation
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
            action_threshold=action_threshold,
            lambda_downside=lambda_downside,
            eta_trade_penalty=eta_trade_penalty,
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
                action_threshold=action_threshold,
                lambda_downside=lambda_downside,
                eta_trade_penalty=eta_trade_penalty,
            )

        # Buy & Hold baseline
        test_prices = test_df["Close"].values
        buy_hold_curve = float(initial_balance) * (test_prices / test_prices[0])

        benchmark_df = None
        if run_benchmark_suite and performance_mode == "Full mode":
            with st.spinner("Running benchmark comparison (B&H / Momentum / Vanilla PPO)..."):
                env_kwargs = {
                    "initial_balance": float(initial_balance),
                    "trade_fee": trade_fee,
                    "slippage_bps": slippage_bps,
                    "spread_bps": spread_bps,
                    "maker_fee": maker_fee,
                    "taker_fee": taker_fee,
                    "min_trade_pct": min_trade_pct,
                    "min_notional": min_notional,
                    "min_qty": min_qty,
                    "qty_step": qty_step,
                    "price_step": price_step,
                    "position_step": float(position_step),
                    "slippage_vol_multiplier": slippage_vol_multiplier,
                    "max_drawdown_limit": max_drawdown_limit,
                    "daily_loss_limit": daily_loss_limit,
                    "volatility_target": volatility_target,
                    "action_threshold": action_threshold,
                    "lambda_downside": lambda_downside,
                    "eta_trade_penalty": eta_trade_penalty,
                    "allow_short": True,
                }
                _, vanilla_metrics, _, _, _ = run_ppo_baseline(
                    train_df=train_df,
                    test_df=test_df,
                    feature_cols=vanilla_feature_cols,
                    env_kwargs=env_kwargs,
                    total_timesteps=min(effective_timesteps, 80_000),
                    lr=1e-4,
                    seed=99,
                    use_attention=False,
                )
                momentum_metrics = evaluate_momentum_baseline(test_df, initial_balance=float(initial_balance), fee_rate=trade_fee)
                buy_hold_metrics = compute_metrics(list(buy_hold_curve[: len(equity_curve)]))

                benchmark_df = pd.DataFrame(
                    [
                        {"Model": "SMC-PPO", "CumulativeReturn": metrics["cumulative_return"], "Sharpe": metrics["sharpe_ratio"], "Sortino": metrics.get("sortino_ratio", 0.0), "MaxDrawdown": metrics["max_drawdown"]},
                        {"Model": "Vanilla PPO", "CumulativeReturn": vanilla_metrics["cumulative_return"], "Sharpe": vanilla_metrics["sharpe_ratio"], "Sortino": vanilla_metrics.get("sortino_ratio", 0.0), "MaxDrawdown": vanilla_metrics["max_drawdown"]},
                        {"Model": "Momentum(MACD+MA)", "CumulativeReturn": momentum_metrics["cumulative_return"], "Sharpe": momentum_metrics["sharpe_ratio"], "Sortino": momentum_metrics.get("sortino_ratio", 0.0), "MaxDrawdown": momentum_metrics["max_drawdown"]},
                        {"Model": "Buy&Hold", "CumulativeReturn": buy_hold_metrics["cumulative_return"], "Sharpe": buy_hold_metrics["sharpe_ratio"], "Sortino": buy_hold_metrics.get("sortino_ratio", 0.0), "MaxDrawdown": buy_hold_metrics["max_drawdown"]},
                    ]
                )

        time_axis, use_datetime = get_time_axis(test_df, len(equity_curve))

        # 6) Metrics display
        st.subheader("📊 Test Set Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        cr  = metrics["cumulative_return"]
        bh_cr = (buy_hold_curve[len(equity_curve) - 1] / float(initial_balance)) - 1.0
        m1.metric("Cumulative Return (RL)",  f"{cr * 100:.2f} %", delta=f"{(cr - bh_cr) * 100:.2f} % vs B&H")
        m2.metric("Sharpe Ratio",       f"{metrics['sharpe_ratio']:.3f}")
        m3.metric("Max Drawdown",       f"{metrics['max_drawdown'] * 100:.2f} %")
        m4.metric("Final Equity (USD)", f"{equity_curve[-1]:,.2f}")

        # 6-0) Current market regime
        current_regime = str(df["market_regime"].iloc[-1])
        regime_name_map = {
            "bull_trend": "Bull Trend",
            "bear_trend": "Bear Trend",
            "range_bound": "Range Bound",
            "high_volatility": "High Volatility",
        }
        regime_cn = regime_name_map.get(current_regime, current_regime)
        st.subheader("🌦️ Current Market Regime")
        r1, r2, r3 = st.columns(3)
        r1.metric("Regime", regime_cn)
        regime_thresholds = get_regime_thresholds(current_regime, base_threshold, strictness_multiplier)
        r2.metric("Buy threshold", f"{regime_thresholds['buy'] * 100:.1f} %")
        r3.metric("Sell threshold", f"{regime_thresholds['sell'] * 100:.1f} %")

        # 6-1) Next-bar signal (for 1d data, this can be treated as tomorrow's signal)
        next_signal = infer_next_signal(
            model=model,
            df=df,
            feature_cols=feature_cols,
            current_regime=current_regime,
            base_threshold=base_threshold,
            strictness_multiplier=strictness_multiplier,
        )
        st.subheader("🔮 Next Bar Recommendation")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Recommended action", next_signal["label"])
        s2.metric("Confidence", f"{next_signal['confidence'] * 100:.1f} %")
        s3.metric("Buy score", f"{next_signal['scores'][1] * 100:.1f} %")
        s4.metric("Sell score", f"{next_signal['scores'][0] * 100:.1f} %")
        st.caption(
            f"Raw continuous action: {next_signal['raw_action']:.3f} ({next_signal['raw_label']}) | "
            f"Regime thresholds Buy>={next_signal['thresholds']['buy'] * 100:.1f}% / "
            f"Sell>={next_signal['thresholds']['sell'] * 100:.1f}%"
        )

        if benchmark_df is not None:
            st.subheader("🧭 Benchmark Comparison")
            st.dataframe(benchmark_df, use_container_width=True)

        if run_shap_analysis and performance_mode == "Full mode":
            st.subheader("🔬 SMC Feature Contribution (SHAP / Permutation)")
            obs_matrix = test_env.features
            shap_df = explain_actor_with_shap(model, obs_matrix, feature_cols, max_samples=180)
            st.dataframe(shap_df.head(20), use_container_width=True)

        # 6-2) Advanced risk metrics
        adv = compute_advanced_metrics(equity_curve, action_history)
        st.subheader("🧪 Advanced Risk Metrics")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Sortino Ratio", f"{adv['sortino_ratio']:.3f}")
        a2.metric("Calmar Ratio", f"{adv['calmar_ratio']:.3f}")
        a3.metric("Trade count", f"{adv['trade_count']}")
        a4.metric("Trade density", f"{adv['trade_density'] * 100:.1f} %")

        if stress_df is not None:
            st.subheader("🧱 Cost Stress Test")
            st.dataframe(stress_df, use_container_width=True)
            stress_chart = stress_df.set_index("Scenario")[["FinalNetWorth"]]
            st.bar_chart(stress_chart)

        st.subheader("🧾 Trade Execution Log")
        if len(trade_log) > 0:
            trade_df = pd.DataFrame(trade_log)
            st.dataframe(trade_df.tail(100), use_container_width=True)
            exec_mix = trade_df["execution"].value_counts().rename_axis("Execution").to_frame("Count")
            st.bar_chart(exec_mix)
        else:
            st.info("No trades were recorded in this run (they may have been filtered by risk controls or thresholds).")

        if yf_interval == "1d":
            st.info("This signal maps to the next daily bar (roughly tomorrow's recommendation).")
        else:
            st.info("This signal maps to the next bar (you are currently using a 1h interval).")

        # 7) Charts
        st.subheader("🧩 Historical Performance Dashboard")
        st.pyplot(plot_performance_dashboard(equity_curve, buy_hold_curve, time_axis, use_datetime, action_history))

        st.subheader("📈 Equity Curve")
        st.pyplot(plot_equity_curve(equity_curve, buy_hold_curve, time_axis, use_datetime))

        st.subheader("🔔 Trade Signals")
        st.pyplot(plot_price_with_signals(test_df, action_history, time_axis, use_datetime))

        st.subheader("🗺️ Regime Background View")
        st.pyplot(plot_price_with_regime_overlay(test_df, action_history, time_axis, use_datetime))

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🎯 Action Distribution")
            st.pyplot(plot_action_distribution(action_history))

        with col_b:
            st.subheader("📉 Equity Curve (Data)")
            eq_df = pd.DataFrame({
                "RL Agent": equity_curve,
                "Buy & Hold": buy_hold_curve[: len(equity_curve)],
            })
            if use_datetime:
                eq_df["Datetime"] = time_axis
                eq_df = eq_df.set_index("Datetime")
            st.line_chart(eq_df)

        effective_walk_forward = enable_walk_forward and performance_mode == "Full mode"
        if enable_walk_forward and performance_mode == "Fast mode":
            st.info("Walk-forward is automatically skipped in fast mode to reduce wait time.")

        if effective_walk_forward:
            with st.spinner("Running walk-forward rolling backtest..."):
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
                    action_threshold=action_threshold,
                    lambda_downside=lambda_downside,
                    eta_trade_penalty=eta_trade_penalty,
                )

            if not wf_df.empty:
                st.subheader("🔁 Walk-forward Backtest Results")
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
                st.warning("Walk-forward parameters exceed the data length. Reduce the train/test windows or the number of folds.")

        # 8) Training data summary
        with st.expander("📋 Raw Data Summary"):
            st.dataframe(df_raw.tail(50), use_container_width=True)

    except Exception as e:
        st.error(f"❌ An error occurred:\n\n`{str(e)}`\n\nPlease check the parameters or data and try again.")
        import traceback
        with st.expander("🔧 Detailed Error Message"):
            st.code(traceback.format_exc())

else:
    # Help page
    st.info("👈 Configure the settings in the sidebar, then press 'Start Download & Train' to run the full workflow.")

    st.subheader("📌 System Architecture")
    st.markdown("""
| Module | Description |
|------|------|
| **Data source** | Download BTC-USD history from yfinance or upload your own CSV |
| **Feature engineering** | Technical indicators + SMC (BOS/FVG/Liquidity Sweep) + MTF (1H/4H) causal alignment |
| **Trading environment** | Custom Gymnasium env, continuous action $A_t \in [-1,1]$ with threshold-based execution |
| **RL algorithm** | Stable-Baselines3 **PPO** (Actor-Critic + attention extractor) |
| **Evaluation metrics** | Cumulative return, Sharpe Ratio, max drawdown, and comparison against Buy & Hold |
""")

    st.subheader("🔄 PPO Overview")
    st.markdown("""
**Proximal Policy Optimization (PPO)** is an on-policy Actor-Critic reinforcement learning algorithm:

1. **Actor** outputs the probability of choosing each action in the current state
2. **Critic** estimates the expected return $V(s)$ of the current state
3. **Clipped surrogate objective** limits the update size so the policy does not drift too far from the old policy:

$$L^{CLIP}(\\theta) = \\mathbb{E}_t \\left[ \\min\\left( r_t(\\theta) \\hat{A}_t,\\ \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t \\right) \\right]$$

where $r_t(\\theta) = \\dfrac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{old}}(a_t|s_t)}$ and $\\hat{A}_t$ is the GAE advantage estimate.
""")
