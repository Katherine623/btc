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
    run_btn = st.button("🚀 開始下載 & 訓練", use_container_width=True)

# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
CSV_PATH = "btc_usdt_1h.csv"
MODEL_PATH = "ppo_btc_trading_agent"


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
                df_raw = download_btc_data(
                    symbol="BTC-USD",
                    interval=yf_interval,
                    period=yf_period,
                    save_path=CSV_PATH,
                    max_retries=3,
                )
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


def plot_equity_curve(equity_curve, buy_hold_curve, title="Equity Curve"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_curve, label="RL Agent", color="#f7931a", linewidth=1.5)
    ax.plot(
        buy_hold_curve[: len(equity_curve)],
        label="Buy & Hold",
        color="#4c72b0",
        linewidth=1.5,
        linestyle="--",
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
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


def plot_price_with_signals(test_df, action_history):
    prices = test_df["Close"].values[: len(action_history)]
    steps = list(range(len(action_history)))

    buy_steps  = [i for i, a in enumerate(action_history) if a == 1]
    sell_steps = [i for i, a in enumerate(action_history) if a == 2]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, prices, color="gray", linewidth=1, label="Close Price")
    ax.scatter(buy_steps,  [prices[i] for i in buy_steps],  marker="^", color="green",  s=60, zorder=5, label="Buy")
    ax.scatter(sell_steps, [prices[i] for i in sell_steps], marker="v", color="red",    s=60, zorder=5, label="Sell")
    ax.set_title("Price Chart with Trade Signals")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("BTC Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# 執行
# ──────────────────────────────────────────────
if run_btn:
    try:
        # 1) 資料
        df_raw = load_or_download_data()

        with st.spinner("計算技術指標..."):
            df = add_technical_indicators(df_raw)
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
            )

        train_env = DummyVecEnv([make_train_env])

        # 4) 訓練 PPO
        with st.spinner(f"訓練 PPO 模型中（{total_timesteps:,} 步）..."):
            # 增大網絡容量以提升學習能力
            policy_kwargs = dict(
                net_arch=[256, 256, 128],  # 更深的網絡
            )
            
            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=5e-4,  # 提高學習率以加快收斂
                n_steps=1024,  # 降低以增加更新頻率
                batch_size=32,  # 更小的批量以增加梯度更新
                n_epochs=15,  # 增加 epoch 以更充分利用數據
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.02,  # 增加熵正則化以鼓勵探索
                verbose=0,
                device="auto",
                policy_kwargs=policy_kwargs,
                seed=42,  # 固定隨機種子以增加可重複性
            )
            model.learn(total_timesteps=total_timesteps)
            model.save(MODEL_PATH)

        st.success(f"✅ 模型訓練完成，已儲存為 `{MODEL_PATH}.zip`")

        # 5) 測試評估
        test_env = BitcoinTradingEnv(
            df=test_df,
            feature_cols=feature_cols,
            initial_balance=float(initial_balance),
            trade_fee=trade_fee,
        )
        metrics, equity_curve, action_history = evaluate_agent(model, test_env)

        # Buy & Hold baseline
        test_prices = test_df["Close"].values
        buy_hold_curve = float(initial_balance) * (test_prices / test_prices[0])

        # 6) 指標顯示
        st.subheader("📊 測試集績效指標")
        m1, m2, m3, m4 = st.columns(4)
        cr  = metrics["cumulative_return"]
        bh_cr = (buy_hold_curve[len(equity_curve) - 1] / float(initial_balance)) - 1.0
        m1.metric("累積報酬率（RL）",  f"{cr * 100:.2f} %", delta=f"{(cr - bh_cr) * 100:.2f} % vs B&H")
        m2.metric("Sharpe Ratio",       f"{metrics['sharpe_ratio']:.3f}")
        m3.metric("最大回撤",            f"{metrics['max_drawdown'] * 100:.2f} %")
        m4.metric("最終資產（USD）",     f"{equity_curve[-1]:,.2f}")

        # 7) 圖表
        st.subheader("📈 資產曲線")
        st.pyplot(plot_equity_curve(equity_curve, buy_hold_curve))

        st.subheader("🔔 交易訊號")
        st.pyplot(plot_price_with_signals(test_df, action_history))

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
            st.line_chart(eq_df)

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
