# Bitcoin RL Trading with PPO

這是一個以比特幣交易為主題的強化學習專案，使用 Stable-Baselines3 的 PPO 演算法，訓練代理人根據歷史價格與技術指標進行買入、賣出與持有決策。

專案同時提供兩種使用方式：

- 命令列訓練腳本：直接下載資料、訓練模型、評估績效並繪圖
- Streamlit 互動介面：調整資料來源與訓練參數，視覺化呈現績效與交易訊號

## 專案特色

- 使用 PPO 訓練 Bitcoin 交易代理人
- 支援從 yfinance 自動下載 BTC-USD 歷史資料
- 計算多個常見技術指標作為觀察特徵
- 提供自訂 Gymnasium 交易環境
- 比較 RL 策略與 Buy and Hold 基準績效
- 透過 Streamlit 顯示資產曲線、動作分佈與交易訊號

## 專案架構

```text
.
├── app_btc.py
├── btc_rl_trading_ppo.py
├── report.md
└── README.md
```

檔案說明：

- `btc_rl_trading_ppo.py`：核心訓練程式，包含資料下載、特徵工程、交易環境、PPO 訓練與回測
- `app_btc.py`：Streamlit 互動介面，可直接操作資料下載、訓練與視覺化
- `report.md`：課堂專題報告草稿
- `README.md`：專案說明文件

## 模型流程

整體流程如下：

1. 下載或讀取 BTC 歷史資料
2. 計算技術指標並建立特徵欄位
3. 將資料切成訓練集與測試集
4. 在自訂交易環境中訓練 PPO 模型
5. 使用測試集評估模型績效
6. 與 Buy and Hold 基準策略比較

## 特徵工程

目前模型使用以下 13 個特徵：

- `return_1`
- `ma_5`
- `ma_10`
- `ma_20`
- `volatility_10`
- `rsi_14`
- `macd`
- `macd_signal`
- `macd_hist`
- `close_over_ma5`
- `close_over_ma10`
- `close_over_ma20`
- `vol_ratio`

這些特徵會在訓練前做標準化，並和代理人本身狀態一起組成 observation。

## 交易環境設計

`BitcoinTradingEnv` 為一個離散動作交易環境。

### 動作空間

- `0`：Hold
- `1`：Buy，若目前為現金則全倉買入 BTC
- `2`：Sell，若目前持有 BTC 則全部賣出

### 持倉狀態

- `0`：空倉
- `1`：滿倉持有 BTC

### Observation 組成

每一個 observation 由兩部分組成：

- 技術指標特徵
- 代理人自身狀態

代理人自身狀態包含：

- 當前持倉狀態
- 現金餘額相對初始資金的比例
- 淨值相對初始資金的比例

### Reward 設計

reward 使用投資組合淨值變化率表示：

```text
reward = (current_net_worth - old_net_worth) / old_net_worth
```

交易手續費會在買入與賣出時計入，因此 reward 會間接反映交易成本。

## 評估指標

測試階段會輸出以下指標：

- 累積報酬率 `cumulative_return`
- 夏普值 `sharpe_ratio`
- 最大回撤 `max_drawdown`
- 最終資產淨值 `final net worth`

同時也會與 Buy and Hold 策略進行比較。

## 安裝方式

建議使用虛擬環境。

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install "stable-baselines3[extra]" gymnasium yfinance streamlit matplotlib pandas numpy
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "stable-baselines3[extra]" gymnasium yfinance streamlit matplotlib pandas numpy
```

## 執行方式

### 1. 直接執行訓練腳本

```powershell
python btc_rl_trading_ppo.py
```

這個腳本會：

- 預設下載或讀取 `btc_usdt_1h.csv`
- 建立特徵與環境
- 訓練 PPO 模型
- 儲存模型為 `ppo_btc_trading_agent.zip`
- 顯示測試績效與資產曲線

### 2. 啟動 Streamlit 互動介面

```powershell
python -m streamlit run app_btc.py
```

在介面中可以：

- 選擇從 yfinance 下載資料或上傳 CSV
- 設定下載期間與 K 棒週期
- 調整初始資金、手續費、訓練步數與訓練集比例
- 觀察績效指標、資產曲線與交易訊號

## 資料格式

若使用自備 CSV，需至少包含以下欄位：

```text
Datetime, Open, High, Low, Close, Volume
```

欄位名稱不必完全一致，程式會自動處理常見變形，例如：

- `datetime`
- `date`
- `timestamp`

## 主要超參數

PPO 預設訓練參數如下：

```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
```

這些參數目前設計為課堂專題與示範用途，若要提升策略品質，可以進一步進行參數搜尋與回測驗證。

## 輸出結果

執行後常見輸出包含：

- `btc_usdt_1h.csv`：下載後的 BTC 歷史資料
- `ppo_btc_trading_agent.zip`：訓練後的 PPO 模型
- Matplotlib 圖表：命令列模式下顯示資產曲線
- Streamlit 圖表：網頁模式下顯示績效與交易訊號

## 適用情境

這個專案適合：

- 課堂專題展示
- 強化學習入門實作
- 金融交易環境建模練習
- PPO 在時間序列決策問題上的基礎示範

## 限制與注意事項

- 目前環境僅支援全倉買入與全倉賣出，未支援部分倉位控制
- reward 設計偏向教學與展示用途，不代表真實交易最佳化目標
- 未納入滑價、流動性限制與更完整的風險控制
- 單次結果會受下載資料區間、隨機性與超參數影響
- 本專案不構成任何投資建議

## 未來可擴充方向

- 納入更多技術指標或鏈上資料
- 加入部位比例控制與風險管理機制
- 改用更長期、多商品或多資產配置環境
- 增加 walk-forward validation 與更完整的回測流程
- 比較 PPO、DQN、A2C 等不同 RL 方法

## 授權與說明

本專案主要用於學習、研究與課堂展示用途。
