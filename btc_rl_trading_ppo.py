# btc_rl_trading_ppo.py
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# =========================================================
# 1. Data loading + feature engineering
# =========================================================
def download_btc_data(
    symbol: str = "BTC-USD",
    interval: str = "1h",
    period: str = "2y",
    save_path: str = "btc_usdt_1h.csv",
) -> pd.DataFrame:
    """
    使用 yfinance 下載 BTC 歷史資料並儲存為 CSV。
    interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    period:   1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("請先安裝 yfinance：pip install yfinance") from exc

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"yfinance 無法取得 {symbol} 的資料，請確認 symbol 與參數。")

    df = df.reset_index()
    df = df.rename(columns={"Datetime": "Datetime", "Date": "Datetime"})

    # 保留需要的欄位
    keep_cols = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    df.to_csv(save_path, index=False)
    print(f"資料已儲存至 {save_path}，共 {len(df)} 筆")
    return df


def load_data(csv_path: str) -> pd.DataFrame:
    """
    CSV format example:
    Datetime,Open,High,Low,Close,Volume
    2023-01-01 00:00:00,....
    """
    df = pd.read_csv(csv_path)

    # 自動處理常見欄位名稱
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "datetime" or lc == "date" or lc == "timestamp":
            col_map[c] = "Datetime"
        elif lc == "open":
            col_map[c] = "Open"
        elif lc == "high":
            col_map[c] = "High"
        elif lc == "low":
            col_map[c] = "Low"
        elif lc == "close":
            col_map[c] = "Close"
        elif lc == "volume":
            col_map[c] = "Volume"

    df = df.rename(columns=col_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime").reset_index(drop=True)

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Return
    data["return_1"] = data["Close"].pct_change()

    # Moving averages
    data["ma_5"] = data["Close"].rolling(5).mean()
    data["ma_10"] = data["Close"].rolling(10).mean()
    data["ma_20"] = data["Close"].rolling(20).mean()

    # Volatility
    data["volatility_10"] = data["return_1"].rolling(10).std()

    # RSI(14)
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    data["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    # Price position
    data["close_over_ma5"] = data["Close"] / (data["ma_5"] + 1e-8)
    data["close_over_ma10"] = data["Close"] / (data["ma_10"] + 1e-8)
    data["close_over_ma20"] = data["Close"] / (data["ma_20"] + 1e-8)

    # Volume normalization
    data["vol_ma_20"] = data["Volume"].rolling(20).mean()
    data["vol_ratio"] = data["Volume"] / (data["vol_ma_20"] + 1e-8)

    data = data.dropna().reset_index(drop=True)
    return data


def build_feature_columns() -> list:
    return [
        "return_1",
        "ma_5",
        "ma_10",
        "ma_20",
        "volatility_10",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "close_over_ma5",
        "close_over_ma10",
        "close_over_ma20",
        "vol_ratio",
    ]


# =========================================================
# 2. Trading environment
# =========================================================
class BitcoinTradingEnv(gym.Env):
    """
    Discrete action trading env:
    0 = Hold
    1 = Buy  (target position -> 1)
    2 = Sell (target position -> 0)

    position:
    - 0 = no BTC
    - 1 = fully invested in BTC

    Reward:
    portfolio value change ratio - trading cost
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        initial_balance: float = 10000.0,
        trade_fee: float = 0.001,      # 0.1%
        window_size: int = 1,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.trade_fee = trade_fee
        self.window_size = window_size

        self.prices = self.df["Close"].values.astype(np.float32)
        self.features = self.df[self.feature_cols].values.astype(np.float32)

        # normalize features roughly
        self.feat_mean = self.features.mean(axis=0, keepdims=True)
        self.feat_std = self.features.std(axis=0, keepdims=True) + 1e-8
        self.features = (self.features - self.feat_mean) / self.feat_std

        self.action_space = spaces.Discrete(3)

        obs_dim = self.features.shape[1] + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.reset()

    def _get_observation(self):
        feat = self.features[self.current_step]

        # 額外加入 agent 自身狀態
        agent_state = np.array([
            self.position,  # 0 or 1
            self.balance / self.initial_balance,
            self.net_worth / self.initial_balance,
        ], dtype=np.float32)

        obs = np.concatenate([feat, agent_state], axis=0).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.btc_units = 0.0
        self.position = 0  # 0 cash, 1 fully long BTC
        self.net_worth = float(self.initial_balance)
        self.prev_net_worth = float(self.initial_balance)

        self.net_worth_history = [self.net_worth]
        self.action_history = []

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        action = int(action)
        price = float(self.prices[self.current_step])

        old_net_worth = self.net_worth

        # action handling
        # 0 = hold
        # 1 = buy  -> all in BTC if currently in cash
        # 2 = sell -> liquidate BTC if currently holding
        if action == 1 and self.position == 0:
            # buy all with fee
            spendable = self.balance * (1.0 - self.trade_fee)
            self.btc_units = spendable / price
            self.balance = 0.0
            self.position = 1

        elif action == 2 and self.position == 1:
            # sell all with fee
            proceeds = self.btc_units * price * (1.0 - self.trade_fee)
            self.balance = proceeds
            self.btc_units = 0.0
            self.position = 0

        # update net worth
        if self.position == 1:
            self.net_worth = self.btc_units * price
        else:
            self.net_worth = self.balance

        # reward = portfolio change ratio
        reward = (self.net_worth - old_net_worth) / (old_net_worth + 1e-8)

        self.action_history.append(action)
        self.net_worth_history.append(self.net_worth)

        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros_like(self._get_observation(), dtype=np.float32)

        info = {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "btc_units": self.btc_units,
            "position": self.position,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(
            f"Step: {self.current_step}, "
            f"Price: {self.prices[self.current_step]:.2f}, "
            f"Position: {self.position}, "
            f"Net Worth: {self.net_worth:.2f}"
        )


# =========================================================
# 3. Evaluation metrics
# =========================================================
def compute_metrics(net_worth_history: list):
    equity = np.array(net_worth_history, dtype=np.float64)
    returns = equity[1:] / (equity[:-1] + 1e-8) - 1.0

    cumulative_return = equity[-1] / equity[0] - 1.0

    if returns.std() > 1e-12:
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
    else:
        sharpe = 0.0

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / (running_max + 1e-8)
    max_drawdown = drawdown.min()

    return {
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }


def evaluate_agent(model, env: BitcoinTradingEnv):
    obs, _ = env.reset()

    done = False
    truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    metrics = compute_metrics(env.net_worth_history)
    return metrics, env.net_worth_history, env.action_history


# =========================================================
# 4. Main pipeline
# =========================================================
def main():
    # =====================================================
    # 修改成你的 BTC CSV 路徑
    # =====================================================
    csv_path = "btc_usdt_1h.csv"

    # 1) load data
    df = load_data(csv_path)
    df = add_technical_indicators(df)
    feature_cols = build_feature_columns()

    # 2) split train / test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print("Total samples:", len(df))
    print("Train samples:", len(train_df))
    print("Test samples :", len(test_df))

    # 3) build env
    def make_train_env():
        return BitcoinTradingEnv(
            df=train_df,
            feature_cols=feature_cols,
            initial_balance=10000.0,
            trade_fee=0.001,
        )

    train_env = DummyVecEnv([make_train_env])

    test_env = BitcoinTradingEnv(
        df=test_df,
        feature_cols=feature_cols,
        initial_balance=10000.0,
        trade_fee=0.001,
    )

    # 4) train PPO
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="auto",
    )

    model.learn(total_timesteps=100_000)
    model.save("ppo_btc_trading_agent")

    # 5) evaluate on test
    metrics, equity_curve, action_history = evaluate_agent(model, test_env)

    print("\n===== Test Metrics =====")
    print(f"Cumulative Return : {metrics['cumulative_return']:.4f}")
    print(f"Sharpe Ratio      : {metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown      : {metrics['max_drawdown']:.4f}")
    print(f"Final Net Worth   : {equity_curve[-1]:.2f}")

    # 6) buy-and-hold baseline
    test_prices = test_df["Close"].values
    buy_hold_curve = 10000.0 * (test_prices / test_prices[0])

    # 7) plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="RL Agent Equity")
    plt.plot(buy_hold_curve[:len(equity_curve)], label="Buy & Hold")
    plt.title("Bitcoin RL Trading Equity Curve")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()