# btc_rl_trading_ppo.py
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv


# =========================================================
# 1. Data loading + causal SMC/MTF feature engineering
# =========================================================
def download_btc_data(
    symbol: str = "BTC-USD",
    interval: str = "1h",
    period: str = "2y",
    save_path: str = "btc_usdt_1h.csv",
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    使用 yfinance 下載 BTC 歷史資料並儲存為 CSV（含重試機制）。
    interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    period:   1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("請先安裝 yfinance：pip install yfinance") from exc

    import time

    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise ValueError(f"yfinance 無法取得 {symbol} 的資料，請確認 symbol 與參數。")

            df = df.reset_index()
            df = df.rename(columns={"Datetime": "Datetime", "Date": "Datetime"})

            keep_cols = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in keep_cols if c in df.columns]]

            df.to_csv(save_path, index=False)
            print(f"資料已儲存至 {save_path}，共 {len(df)} 筆")
            return df

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"下載失敗（嘗試 {attempt + 1}/{max_retries}）：{str(e)[:100]}")
                print(f"等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
            else:
                raise Exception(f"下載 {symbol} 失敗（嘗試 {max_retries} 次）：{str(e)}") from e


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("datetime", "date", "timestamp"):
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


def _causal_smooth(series: pd.Series, weights: Tuple[float, ...] = (0.6, 0.3, 0.1)) -> pd.Series:
    w = np.array(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-8)
    out = pd.Series(np.zeros(len(series)), index=series.index, dtype=np.float64)
    for idx, weight in enumerate(w):
        out += weight * series.shift(idx).bfill().fillna(0.0)
    return out


def _compute_smc_columns(data: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    out = data.copy()

    swing_lookback = 12
    liq_lookback = 20

    prev_swing_high = out["High"].rolling(swing_lookback).max().shift(1)
    prev_swing_low = out["Low"].rolling(swing_lookback).min().shift(1)

    out[f"{prefix}bos_up"] = (
        (out["Close"] > prev_swing_high)
        & (out["Close"].shift(1) <= prev_swing_high.shift(1))
    ).astype(float)
    out[f"{prefix}bos_down"] = (
        (out["Close"] < prev_swing_low)
        & (out["Close"].shift(1) >= prev_swing_low.shift(1))
    ).astype(float)

    out[f"{prefix}fvg_gap"] = (out["High"] - out["Low"].shift(2)) / (out["Close"] + 1e-8)

    recent_liq_high = out["High"].rolling(liq_lookback).max().shift(1)
    recent_liq_low = out["Low"].rolling(liq_lookback).min().shift(1)
    out[f"{prefix}liq_sweep_high"] = (
        (out["High"] > recent_liq_high) & (out["Close"] < recent_liq_high)
    ).astype(float)
    out[f"{prefix}liq_sweep_low"] = (
        (out["Low"] < recent_liq_low) & (out["Close"] > recent_liq_low)
    ).astype(float)

    range_high = out["High"].rolling(liq_lookback).max().shift(1)
    range_low = out["Low"].rolling(liq_lookback).min().shift(1)
    pd_ratio = (out["Close"] - range_low) / (range_high - range_low + 1e-8)
    out[f"{prefix}premium_discount"] = pd_ratio.clip(0.0, 1.0)

    out[f"{prefix}smc_score"] = (
        1.6 * out[f"{prefix}bos_up"]
        - 1.2 * out[f"{prefix}bos_down"]
        + 0.8 * out[f"{prefix}liq_sweep_low"]
        - 0.8 * out[f"{prefix}liq_sweep_high"]
        + 0.4 * (out[f"{prefix}premium_discount"] - 0.5)
        + 0.2 * out[f"{prefix}fvg_gap"].clip(-0.05, 0.05)
    )

    out[f"{prefix}smc_score"] = _causal_smooth(out[f"{prefix}smc_score"])
    return out


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlcv = (
        df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]
        .resample(rule, label="right", closed="right")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    return ohlcv


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Base technical features
    data["return_1"] = data["Close"].pct_change()
    data["ma_5"] = data["Close"].rolling(5).mean()
    data["ma_10"] = data["Close"].rolling(10).mean()
    data["ma_20"] = data["Close"].rolling(20).mean()
    data["volatility_10"] = data["return_1"].rolling(10).std()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    data["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    data["close_over_ma5"] = data["Close"] / (data["ma_5"] + 1e-8)
    data["close_over_ma10"] = data["Close"] / (data["ma_10"] + 1e-8)
    data["close_over_ma20"] = data["Close"] / (data["ma_20"] + 1e-8)

    data["vol_ma_20"] = data["Volume"].rolling(20).mean()
    data["vol_ratio"] = data["Volume"] / (data["vol_ma_20"] + 1e-8)

    # Causal SMC features at base timeframe
    data = _compute_smc_columns(data, prefix="")

    # MTF alignment (only if datetime exists)
    if "Datetime" in data.columns:
        base = data.copy()

        h1 = _resample_ohlcv(base, "1h")
        h1 = _compute_smc_columns(h1, prefix="h1_")
        h1_feats = [
            "Datetime",
            "h1_bos_up",
            "h1_bos_down",
            "h1_fvg_gap",
            "h1_liq_sweep_high",
            "h1_liq_sweep_low",
            "h1_premium_discount",
            "h1_smc_score",
        ]
        base = pd.merge_asof(
            base.sort_values("Datetime"),
            h1[h1_feats].sort_values("Datetime"),
            on="Datetime",
            direction="backward",
        )

        h4 = _resample_ohlcv(base[["Datetime", "Open", "High", "Low", "Close", "Volume"]], "4h")
        h4 = _compute_smc_columns(h4, prefix="h4_")
        h4_feats = [
            "Datetime",
            "h4_bos_up",
            "h4_bos_down",
            "h4_fvg_gap",
            "h4_liq_sweep_high",
            "h4_liq_sweep_low",
            "h4_premium_discount",
            "h4_smc_score",
        ]
        base = pd.merge_asof(
            base.sort_values("Datetime"),
            h4[h4_feats].sort_values("Datetime"),
            on="Datetime",
            direction="backward",
        )

        data = base

    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return data


def build_vanilla_feature_columns() -> List[str]:
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


def build_feature_columns() -> List[str]:
    base_cols = build_vanilla_feature_columns()
    smc_cols = [
        "bos_up",
        "bos_down",
        "fvg_gap",
        "liq_sweep_high",
        "liq_sweep_low",
        "premium_discount",
        "smc_score",
        "h1_bos_up",
        "h1_bos_down",
        "h1_fvg_gap",
        "h1_liq_sweep_high",
        "h1_liq_sweep_low",
        "h1_premium_discount",
        "h1_smc_score",
        "h4_bos_up",
        "h4_bos_down",
        "h4_fvg_gap",
        "h4_liq_sweep_high",
        "h4_liq_sweep_low",
        "h4_premium_discount",
        "h4_smc_score",
    ]
    return base_cols + smc_cols


# =========================================================
# 2. Attention-based feature extractor (Actor-Critic core)
# =========================================================
class SMCAttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    用 feature-wise attention 對市場特徵加權，讓 policy 自動聚焦關鍵結構訊號。
    observation = [market_features..., agent_state]
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        market_feature_dim: int,
        agent_state_dim: int = 4,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        self.market_feature_dim = market_feature_dim
        self.agent_state_dim = agent_state_dim

        self.agent_proj = nn.Sequential(
            nn.Linear(agent_state_dim, market_feature_dim),
            nn.Tanh(),
        )

        self.combiner = nn.Sequential(
            nn.Linear(market_feature_dim + agent_state_dim, 192),
            nn.ReLU(),
            nn.Linear(192, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        market = observations[:, : self.market_feature_dim]
        agent = observations[:, self.market_feature_dim : self.market_feature_dim + self.agent_state_dim]

        context = self.agent_proj(agent)
        scores = th.tanh(market * context)
        attn = th.softmax(scores, dim=1)
        attended_market = market * attn

        return self.combiner(th.cat([attended_market, agent], dim=1))


def build_attention_policy_kwargs(market_feature_dim: int, agent_state_dim: int = 4) -> Dict:
    return {
        "features_extractor_class": SMCAttentionFeatureExtractor,
        "features_extractor_kwargs": {
            "market_feature_dim": market_feature_dim,
            "agent_state_dim": agent_state_dim,
            "features_dim": 128,
        },
        "net_arch": {
            "pi": [128, 128],
            "vf": [128, 128],
        },
    }


# =========================================================
# 3. Continuous-action event-driven trading environment
# =========================================================
class BitcoinTradingEnv(gym.Env):
    """
    Continuous action env:
    action in [-1, 1] means target capital exposure.
    -1 = fully short, +1 = fully long, 0 = flat.

    Action threshold:
    if |delta_action| <= threshold, no real trade is executed.
    A tiny penalty is still applied to keep policy gradients continuous.

    Reward:
    R_t = ΔPnL_t - λ * DownsideDeviation - η * Penalty_trade
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        initial_balance: float = 10000.0,
        trade_fee: float = 0.001,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
        slippage_bps: float = 8.0,
        spread_bps: float = 4.0,
        min_trade_pct: float = 0.02,
        position_step: float = 0.25,
        slippage_vol_multiplier: float = 1.2,
        min_notional: float = 10.0,
        min_qty: float = 0.0001,
        qty_step: float = 0.0001,
        price_step: float = 0.01,
        max_drawdown_limit: float = 0.30,
        daily_loss_limit: float = 0.06,
        volatility_target: float = 0.02,
        action_threshold: float = 0.10,
        downside_window: int = 24,
        lambda_downside: float = 0.12,
        eta_trade_penalty: float = 0.004,
        tiny_trade_penalty: float = 0.0007,
        allow_short: bool = True,
        max_leverage: float = 1.0,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = feature_cols
        self.initial_balance = float(initial_balance)

        self.trade_fee = float(trade_fee)
        self.maker_fee = float(maker_fee)
        self.taker_fee = float(taker_fee)
        self.slippage_bps = float(slippage_bps)
        self.spread_bps = float(spread_bps)
        self.min_trade_pct = float(min_trade_pct)
        self.position_step = float(position_step)
        self.slippage_vol_multiplier = float(slippage_vol_multiplier)
        self.min_notional = float(min_notional)
        self.min_qty = float(min_qty)
        self.qty_step = float(qty_step)
        self.price_step = float(price_step)
        self.max_drawdown_limit = float(max_drawdown_limit)
        self.daily_loss_limit = float(daily_loss_limit)
        self.volatility_target = float(volatility_target)
        self.action_threshold = float(action_threshold)
        self.downside_window = int(max(4, downside_window))
        self.lambda_downside = float(lambda_downside)
        self.eta_trade_penalty = float(eta_trade_penalty)
        self.tiny_trade_penalty = float(tiny_trade_penalty)
        self.allow_short = bool(allow_short)
        self.max_leverage = float(max(0.5, max_leverage))

        self.prices = self.df["Close"].values.astype(np.float32)
        self.features_raw = self.df[self.feature_cols].values.astype(np.float32)

        self.feat_mean = self.features_raw.mean(axis=0, keepdims=True)
        self.feat_std = self.features_raw.std(axis=0, keepdims=True) + 1e-8
        self.features = (self.features_raw - self.feat_mean) / self.feat_std

        if "volatility_10" in self.df.columns:
            self.volatility_series = self.df["volatility_10"].fillna(0.0).values.astype(np.float32)
        else:
            self.volatility_series = np.zeros(len(self.df), dtype=np.float32)
        self.vol_median = float(np.median(self.volatility_series) + 1e-8)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        obs_dim = self.features.shape[1] + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.reset()

    def _get_observation(self) -> np.ndarray:
        feat = self.features[self.current_step]
        agent_state = np.array(
            [
                self.exposure,
                self.balance / (self.initial_balance + 1e-8),
                self.net_worth / (self.initial_balance + 1e-8),
                self.last_drawdown,
            ],
            dtype=np.float32,
        )
        return np.concatenate([feat, agent_state], axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.net_worth = float(self.initial_balance)
        self.peak_net_worth = float(self.initial_balance)
        self.episode_start_net_worth = float(self.initial_balance)

        self.exposure = 0.0
        self.last_drawdown = 0.0
        self.recent_pnl_returns: List[float] = []

        self.net_worth_history = [self.net_worth]
        self.action_history: List[int] = []
        self.raw_action_history: List[float] = []
        self.exposure_history: List[float] = [self.exposure]
        self.turnover_history: List[float] = [0.0]
        self.trade_log: List[Dict] = []

        return self._get_observation(), {}

    def _dynamic_cost_rate(self, trade_size: float, vol_now: float, net_worth: float) -> Tuple[float, str]:
        vol_scale = vol_now / self.vol_median
        dynamic_slippage = (self.slippage_bps / 10000.0) * (1.0 + self.slippage_vol_multiplier * vol_scale)
        spread_cost = (self.spread_bps / 10000.0) * 0.5

        maker_vs_taker_score = min(1.0, trade_size / (self.min_trade_pct + 1e-8))
        maker_weight = float(np.clip(1.0 - maker_vs_taker_score, 0.0, 1.0))
        execution_fee = maker_weight * self.maker_fee + (1.0 - maker_weight) * self.taker_fee
        execution_type = "maker" if maker_weight >= 0.5 else "taker"

        total_cost_rate = self.trade_fee + execution_fee + dynamic_slippage + spread_cost
        return float(total_cost_rate), execution_type

    def step(self, action):
        raw_action = float(np.asarray(action).reshape(-1)[0])
        raw_action = float(np.clip(raw_action, -1.0, 1.0))
        if not self.allow_short:
            raw_action = max(0.0, raw_action)

        current_price = float(self.prices[self.current_step])
        next_price = float(self.prices[min(self.current_step + 1, len(self.prices) - 1)])
        if self.price_step > 0:
            current_price = np.floor(current_price / self.price_step) * self.price_step
            next_price = np.floor(next_price / self.price_step) * self.price_step

        old_net_worth = float(self.net_worth)
        vol_now = float(self.volatility_series[self.current_step])

        # Volatility targeting shrinks exposure in stressed conditions.
        vol_scale_target = self.volatility_target / (vol_now + 1e-8) if vol_now > 0 else 1.0
        risk_scale = float(np.clip(vol_scale_target, 0.25, 1.0))
        target_exposure = float(np.clip(raw_action * risk_scale, -self.max_leverage, self.max_leverage))
        if not self.allow_short:
            target_exposure = max(0.0, target_exposure)

        delta_action = target_exposure - self.exposure
        trade_size = abs(delta_action)

        executed_trade = False
        tiny_penalty = 0.0
        trade_penalty = 0.0
        executed_exposure = self.exposure
        execution_type = "none"
        trade_cost_rate = 0.0

        notion_threshold = max(self.min_trade_pct, self.min_notional / (old_net_worth + 1e-8))
        if trade_size <= self.action_threshold or trade_size <= notion_threshold:
            tiny_penalty = self.tiny_trade_penalty * (trade_size / (self.action_threshold + 1e-8))
        else:
            trade_cost_rate, execution_type = self._dynamic_cost_rate(trade_size, vol_now, old_net_worth)
            executed_exposure = target_exposure
            trade_penalty = self.eta_trade_penalty * trade_size
            executed_trade = True

        price_return = (next_price - current_price) / (current_price + 1e-8)
        pnl_return = executed_exposure * price_return

        net_return = pnl_return - (trade_cost_rate * trade_size)
        self.net_worth = float(max(10.0, old_net_worth * (1.0 + net_return)))

        self.exposure = executed_exposure
        self.exposure_history.append(self.exposure)

        self.balance = float(max(0.0, self.net_worth * (1.0 - min(0.99, abs(self.exposure)))))
        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)

        drawdown = (self.peak_net_worth - self.net_worth) / (self.peak_net_worth + 1e-8)
        self.last_drawdown = float(drawdown)

        self.recent_pnl_returns.append(float(pnl_return))
        if len(self.recent_pnl_returns) > self.downside_window:
            self.recent_pnl_returns = self.recent_pnl_returns[-self.downside_window :]

        downside = [x for x in self.recent_pnl_returns if x < 0.0]
        downside_deviation = float(np.std(downside)) if len(downside) > 1 else 0.0

        risk_penalty = self.lambda_downside * downside_deviation
        reward = float(pnl_return - risk_penalty - trade_penalty - tiny_penalty)

        day_loss = max(
            0.0,
            (self.episode_start_net_worth - self.net_worth) / (self.episode_start_net_worth + 1e-8),
        )
        risk_breached = drawdown >= self.max_drawdown_limit or day_loss >= self.daily_loss_limit

        if risk_breached:
            reward -= 0.02

        action_signal = 0
        if executed_trade:
            if delta_action > 0:
                action_signal = 1
            elif delta_action < 0:
                action_signal = -1

        self.raw_action_history.append(raw_action)
        self.action_history.append(action_signal)
        self.net_worth_history.append(self.net_worth)
        self.turnover_history.append(trade_size)

        if executed_trade:
            dt_value = ""
            if "Datetime" in self.df.columns:
                dt_value = str(self.df.loc[self.current_step, "Datetime"])
            self.trade_log.append(
                {
                    "step": int(self.current_step),
                    "datetime": dt_value,
                    "raw_action": float(raw_action),
                    "executed_exposure": float(executed_exposure),
                    "execution": execution_type,
                    "trade_size": float(trade_size),
                    "cost_rate": float(trade_cost_rate),
                    "net_worth": float(self.net_worth),
                }
            )

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1 or risk_breached
        truncated = False

        if not terminated:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "exposure": self.exposure,
            "drawdown": drawdown,
            "downside_deviation": downside_deviation,
            "turnover": trade_size,
            "risk_breached": risk_breached,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f"Step: {self.current_step}, "
            f"Price: {self.prices[self.current_step]:.2f}, "
            f"Exposure: {self.exposure:.3f}, "
            f"Net Worth: {self.net_worth:.2f}"
        )


# =========================================================
# 4. Evaluation + baselines
# =========================================================
def compute_metrics(net_worth_history: List[float]) -> Dict[str, float]:
    equity = np.array(net_worth_history, dtype=np.float64)
    returns = equity[1:] / (equity[:-1] + 1e-8) - 1.0

    cumulative_return = float(equity[-1] / equity[0] - 1.0)

    if returns.std() > 1e-12:
        sharpe = float(np.sqrt(252) * returns.mean() / (returns.std() + 1e-8))
    else:
        sharpe = 0.0

    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.0
    sortino = float(np.sqrt(252) * returns.mean() / (downside_std + 1e-8)) if downside_std > 1e-12 else 0.0

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / (running_max + 1e-8)
    max_drawdown = float(drawdown.min())

    return {
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
    }


def evaluate_agent(model, env: BitcoinTradingEnv):
    obs, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)

    metrics = compute_metrics(env.net_worth_history)
    return metrics, env.net_worth_history, env.action_history, env.trade_log


def evaluate_momentum_baseline(
    df: pd.DataFrame,
    initial_balance: float = 10000.0,
    fee_rate: float = 0.001,
) -> Dict[str, float]:
    data = df.copy().reset_index(drop=True)
    signal = np.zeros(len(data), dtype=np.float64)

    ma_fast = data["Close"].rolling(10).mean()
    ma_slow = data["Close"].rolling(30).mean()
    macd_hist = data["macd_hist"] if "macd_hist" in data.columns else pd.Series(np.zeros(len(data)))

    signal[(ma_fast > ma_slow) & (macd_hist > 0)] = 1.0
    signal[(ma_fast < ma_slow) & (macd_hist < 0)] = -1.0

    signal = pd.Series(signal).fillna(0.0).values

    equity = [initial_balance]
    exposure_prev = 0.0
    for i in range(len(data) - 1):
        p0 = float(data.loc[i, "Close"])
        p1 = float(data.loc[i + 1, "Close"])
        ret = (p1 - p0) / (p0 + 1e-8)

        exposure = signal[i]
        cost = abs(exposure - exposure_prev) * fee_rate
        step_ret = exposure * ret - cost

        equity.append(max(10.0, equity[-1] * (1.0 + step_ret)))
        exposure_prev = exposure

    return compute_metrics(equity)


def build_buy_hold_curve(prices: np.ndarray, initial_balance: float) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    return initial_balance * (prices / (prices[0] + 1e-8))


def run_ppo_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    env_kwargs: Dict,
    total_timesteps: int = 100_000,
    lr: float = 1e-4,
    seed: int = 42,
    use_attention: bool = True,
):
    def make_train_env(local_df=train_df):
        return BitcoinTradingEnv(df=local_df, feature_cols=feature_cols, **env_kwargs)

    train_env = DummyVecEnv([make_train_env])
    test_env = BitcoinTradingEnv(df=test_df, feature_cols=feature_cols, **env_kwargs)

    policy_kwargs = None
    if use_attention:
        policy_kwargs = build_attention_policy_kwargs(market_feature_dim=len(feature_cols), agent_state_dim=4)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lr,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        device="auto",
        policy_kwargs=policy_kwargs,
        seed=seed,
    )
    model.learn(total_timesteps=int(total_timesteps))

    metrics, equity_curve, action_history, trade_log = evaluate_agent(model, test_env)
    return model, metrics, equity_curve, action_history, trade_log


# =========================================================
# 5. Optional explainability (SHAP fallback)
# =========================================================
def explain_actor_with_shap(
    model,
    obs_matrix: np.ndarray,
    feature_names: List[str],
    max_samples: int = 256,
) -> pd.DataFrame:
    """
    SHAP for actor mean output. If shap package is missing, fallback to permutation importance.
    """
    obs = np.asarray(obs_matrix, dtype=np.float32)
    if len(obs) == 0:
        return pd.DataFrame(columns=["feature", "importance", "method"])

    idx = np.linspace(0, len(obs) - 1, min(max_samples, len(obs))).astype(int)
    sample = obs[idx]

    def predict_action(x: np.ndarray) -> np.ndarray:
        with th.no_grad():
            tensor_x = th.tensor(x, dtype=th.float32, device=model.device)
            actions = model.policy.predict_values(tensor_x)
        return actions.detach().cpu().numpy().reshape(-1, 1)

    try:
        import shap  # type: ignore

        background = sample[: max(20, min(80, len(sample)))]
        explainer = shap.KernelExplainer(lambda x: predict_action(np.asarray(x, dtype=np.float32)), background)
        shap_values = explainer.shap_values(sample, nsamples=100)
        vals = np.array(shap_values)
        if vals.ndim == 3:
            vals = vals[0]
        importance = np.mean(np.abs(vals), axis=0)
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance[: len(feature_names)],
                "method": "shap",
            }
        ).sort_values("importance", ascending=False)
    except Exception:
        baseline = predict_action(sample).mean()
        importances = []
        for col in range(min(sample.shape[1], len(feature_names))):
            x_perm = sample.copy()
            rng = np.random.default_rng(123 + col)
            rng.shuffle(x_perm[:, col])
            score = np.abs(predict_action(x_perm).mean() - baseline)
            importances.append(float(score))

        return pd.DataFrame(
            {
                "feature": feature_names[: len(importances)],
                "importance": importances,
                "method": "permutation_fallback",
            }
        ).sort_values("importance", ascending=False)


# =========================================================
# 6. Main pipeline demo
# =========================================================
def main():
    csv_path = "btc_usdt_1h.csv"

    df = load_data(csv_path)
    df = add_technical_indicators(df)

    full_features = [c for c in build_feature_columns() if c in df.columns]
    vanilla_features = [c for c in build_vanilla_feature_columns() if c in df.columns]

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print("Total samples:", len(df))
    print("Train samples:", len(train_df))
    print("Test samples :", len(test_df))

    env_kwargs = {
        "initial_balance": 10000.0,
        "trade_fee": 0.001,
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
        "slippage_bps": 8.0,
        "spread_bps": 4.0,
        "action_threshold": 0.10,
        "lambda_downside": 0.12,
        "eta_trade_penalty": 0.004,
        "allow_short": True,
    }

    smc_model, smc_metrics, smc_equity, _, _ = run_ppo_baseline(
        train_df=train_df,
        test_df=test_df,
        feature_cols=full_features,
        env_kwargs=env_kwargs,
        total_timesteps=120_000,
        lr=1e-4,
        seed=42,
        use_attention=True,
    )
    smc_model.save("ppo_btc_trading_agent")

    _, vanilla_metrics, vanilla_equity, _, _ = run_ppo_baseline(
        train_df=train_df,
        test_df=test_df,
        feature_cols=vanilla_features,
        env_kwargs=env_kwargs,
        total_timesteps=120_000,
        lr=1e-4,
        seed=43,
        use_attention=False,
    )

    momentum_metrics = evaluate_momentum_baseline(test_df, initial_balance=10000.0, fee_rate=0.001)

    buy_hold_curve = build_buy_hold_curve(test_df["Close"].values, 10000.0)
    buy_hold_metrics = compute_metrics(list(buy_hold_curve[: len(smc_equity)]))

    print("\n===== Benchmark Summary =====")
    print("SMC-PPO:", smc_metrics)
    print("Vanilla PPO:", vanilla_metrics)
    print("Momentum:", momentum_metrics)
    print("Buy & Hold:", buy_hold_metrics)

    plt.figure(figsize=(12, 6))
    plt.plot(smc_equity, label="SMC-PPO")
    plt.plot(vanilla_equity[: len(smc_equity)], label="Vanilla PPO")
    plt.plot(buy_hold_curve[: len(smc_equity)], label="Buy & Hold", linestyle="--")
    plt.title("Benchmark Equity Curve")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
