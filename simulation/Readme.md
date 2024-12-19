# トレーディング戦略用シミュレーションエンジン

## 概要

`SimulationEngine` は、過去の価格データを使用してトレーディング戦略をシミュレートするためのPythonベースのバックテストフレームワークです。このエンジンは以下の機能を提供します：

- 売買操作の実行とポジション管理
- 残高、手数料、利益の追跡
- 過去の市場データを用いた戦略のテスト
- 詳細な取引履歴と利益履歴の出力

このエンジンは柔軟で、カスタム戦略や手数料構造を組み込むことができます。

## 必要条件

- Python 3.8以上
- 必要ライブラリ: `pandas`

依存関係のインストール：

```bash 
# deb-bot下で以下を実行
pip install -r requirements
```

## 動作の仕組み

1. **初期化:** 初期残高、過去の価格データ、手数料計算関数、必要に応じて市場データ関数を指定します。
2. **市場データ取得:** 戦略ロジック用の過去データを取得します。
3. **取引実行:** 戦略の出力に基づいて売買操作を実行します。
4. **利益計算:** 利益/損失を継続的に追跡し、取引履歴を保存します。

## 使用例

以下は `SimulationEngine` の使用方法の例です。

### 1. エンジンの初期化

```python
import pandas as pd
from simulation import SimulationEngine, fixed_fee, DummyStrategy

# 価格データを読み込む
price_data = pd.read_csv("price_data.csv")

# エンジンを初期化
engine = SimulationEngine(
    initial_balance=10000,  # 初期残高（USD）
    price_data=price_data,
    fee_function=fixed_fee  # 固定手数料: 取引額の0.1%
)

dummy_strategy = DummyStrategy() # 取引ストラテジー
engine.run_simulation(strategy=dummy_strategy, lookback_period=10)
```

### 2. 戦略の定義

戦略は、`market_data`、`balance`、`positions` を入力とし、アクション（`buy`, `sell` など）と取引サイズを返す関数またはオブジェクトとして実装します。

```python
class DummyStrategy:
    """
    A simple strategy for testing:
    Buys when the price is above the 5-period moving average.
    Sells when the price is below the 5-period moving average.
    """
    def __call__(
        self, market_data: pd.DataFrame, balance: Dict[str, float], positions: List[Dict]
    ) -> Tuple[str, float]:
        if market_data.empty:  # Skip if data is insufficient
            return 'hold', 0

        market_data = market_data.copy()  # 明示的にコピーを作成
        market_data['ma'] = market_data['close'].rolling(window=5).mean()
        current_price = market_data.iloc[-1]['close']
        moving_average = market_data.iloc[-1]['ma']

        long_position = next((pos for pos in positions if pos['type'] == 'long'), None)
        short_position = next((pos for pos in positions if pos['type'] == 'short'), None)

        if current_price > moving_average and not long_position:
            return 'buy', balance['cash'] / current_price / 10
        elif current_price < moving_average and long_position:
            return 'close_long', long_position['size']
        elif current_price < moving_average and not short_position:
            return 'sell', balance['cash'] / current_price / 10
        elif current_price > moving_average and short_position:
            return 'close_short', short_position['size']
        return 'hold', 0

strategy = DummyStrategy()
```

### 3. シミュレーションの実行

戦略とオプションでルックバック期間を指定してシミュレーションを実行します。

```python
engine.run_simulation(strategy=strategy, lookback_period=10)
```

### 4. 結果の分析

シミュレーション結果（取引履歴と利益履歴）はCSVファイルに保存されます。

- `trade_history.csv`: 各取引の詳細を含む
- `profit_history.csv`: 時間経過に伴う利益の追跡

## 主な構成要素

### SimulationEngine

シミュレーションを処理するコアクラスです。

#### コンストラクタ:

```python
SimulationEngine(
    initial_balance: float,
    price_data: pd.DataFrame,
    fee_function: Callable[[float], float],
    market_data_function: Optional[Callable[[int, int], pd.DataFrame]] = None
)
```

- `initial_balance`: 初期現金残高。
- `price_data`: DataFrame形式の過去の価格データ。
- `fee_function`: 取引手数料を計算する関数。
- `market_data_function`: 市場データを取得するカスタム関数（オプション）。

#### メソッド:

- `get_market_data(timestamp, lookback_period)`: 市場データを取得。
- `execute_buy(price, size)`: 買い注文を実行。
- `execute_sell(price, size)`: 売り注文を実行。
- `close_position(price, position_type)`: 既存のポジションをクローズ。
- `run_simulation(strategy, lookback_period)`: バックテストシミュレーションを実行。
- `save_results()`: 取引履歴と利益履歴をCSVに保存。

### 手数料関数

手数料ロジックをカスタマイズ可能。例：

```python
def fixed_fee(amount: float) -> float:
    return amount * 0.001  # 0.1% の手数料
```

### 戦略ロジック

カスタム戦略をコール可能なオブジェクトまたは関数として定義します。例：

```python
class DummyStrategy:
    def __call__(self, market_data: pd.DataFrame, balance: Dict[str, float], positions: List[Dict]) -> Tuple[str, float]:
        # 単純移動平均のロジック
        market_data['ma'] = market_data['close'].rolling(window=5).mean()
        current_price = market_data.iloc[-1]['close']
        moving_average = market_data.iloc[-1]['ma']

        if current_price > moving_average:
            return 'buy', balance['cash'] / current_price
        elif current_price < moving_average:
            return 'sell', positions[0]['size'] if positions else 0
        return 'hold', 0
```

### 市場データ関数

デフォルトの市場データ関数は、過去データのスライドウィンドウを取得します。高度な要件にはオーバーライドします。

```python
def custom_market_data_function(timestamp: int, lookback_period: int) -> pd.DataFrame:
    # カスタムロジックを実装
    pass
```

## 戦略条件

独自の戦略を定義する際には、以下を遵守してください：

- **市場データの使用:** 提供された市場データ形式で動作すること。
- **取引サイズ:** 利用可能な残高またはポジションサイズを反映した適切なサイズを返すこと。
- **アクションの出力:** 有効なアクション（`buy`, `sell`, `close_long`, `close_short`, `hold`）を返すこと。

## 出力

エンジンは以下の2つの主要ファイルを生成します：

- **`trade_history.csv`****:**
  - 列: `action`, `price`, `size`, `fee`
- **`profit_history.csv`****:**
  - 列: `timestamp`, `profit`


## テスト
`test_simulation.py`に簡易なテストを実装している。以下コマンドでテストを実施できる。(取引回数が一回のため、シャープレシオ計算の際にpandas内部から警告を吐き出しているので、警告は無視してよい)
```
pytest test_simulation_class.py
```

## 注意点

このエンジンは教育およびテスト目的で設計されています。決定論的な価格データを想定しており、板情報の動態やスリッページを考慮していません。より複雑なバックテストシステムの基盤として活用してください。
