from simulation import SimulationEngine

import pandas as pd
from typing import Callable, List, Dict, Tuple

def fixed_fee(amount):
    return amount * 0.01

# SimulationEngineのテスト
def test_simulation_engine_00():
    # テスト用のシンプルなデータと戦略
    def generate_long_test_data() -> pd.DataFrame:
        """単調に増加するシンプルなテストデータを生成する関数。"""
        data = pd.DataFrame({
            'timestamp': range(10),  # 10個のタイムステップ
            'date': pd.date_range(start='2024-01-01', periods=10, freq='min'),
            'open': [10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900],
            'high': [10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900, 11000],
            'low': [9900, 10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800],
            'close': [10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900],  # 単調増加
        })
        return data

    class LongTestStrategy:
        """テスト用のシンプルな戦略（固定動作）。"""
        def __call__(self, market_data: pd.DataFrame, balance: Dict[str, float], positions: List[Dict]) -> Tuple[str, float]:
            # 現在のタイムステップを取得
            current_step = market_data.iloc[-1]['timestamp']
            if current_step == 1:  # 2番目のタイムステップで買う
                size = 0.1
                return 'buy', size
            elif current_step == 2:  # 次のタイムステップで売る
                return 'close_long', positions[0]['size']
            return 'hold', 0
            
    test_data = generate_long_test_data()
    strategy = LongTestStrategy()

    # 初期化
    engine = SimulationEngine(
        initial_balance=10000,  # 初期資産
        price_data=test_data,
        fee_function=fixed_fee,  # 手数料関数
        output_dir="runs/test_simulation_00"  # 出力を不要にする
    )

    # シミュレーション実行
    final_balance, history, profit_history = engine.run_simulation(strategy, lookback_period=1)

    print(history)
    # 手計算で期待される最終資産を確認
    # 初期価格で10100 * 0.1購入 (10100 * 0.1 = 1010)
    # 次の価格で売却した利益 (10200 - 10100) * 0.1 = 10
    # 手数料 (10100 + 10200) * 0.01 = 203
    expected_profit = 10200* 0.1 - 10100* 0.1 - fixed_fee(10100 * 0.1) - fixed_fee(10200 * 0.1)
    expected_balance = 10000 + expected_profit

    # アサーション
    assert abs(final_balance['cash'] - expected_balance) < 1e-6, f"Test failed: {final_balance['cash']} != {expected_balance}"
    print(f"Test passed: Final balance {final_balance['cash']} matches expected balance {expected_balance}")



def test_simulation_engine_01():
    def generate_short_test_data() -> pd.DataFrame:
        """単調に減少するシンプルなテストデータを生成する関数（ショート用）。"""
        data = pd.DataFrame({
            'timestamp': range(10),  # 10個のタイムステップ
            'date': pd.date_range(start='2024-01-01', periods=10, freq='min'),
            'open': [10000, 9900, 9800, 9700, 9600, 9500, 9400, 9300, 9200, 9100],
            'high': [10100, 10000, 9900, 9800, 9700, 9600, 9500, 9400, 9300, 9200],
            'low': [9900, 9800, 9700, 9600, 9500, 9400, 9300, 9200, 9100, 9000],
            'close': [10000, 9900, 9800, 9700, 9600, 9500, 9400, 9300, 9200, 9100],  # 単調減少
        })
        return data

    class ShortTestStrategy:
        """ショート取引用の戦略（最初に売って、後で買い戻す）。"""
        def __call__(self, market_data: pd.DataFrame, balance: Dict[str, float], positions: List[Dict]) -> Tuple[str, float]:
            # 現在のタイムステップを取得
            current_step = market_data.iloc[-1]['timestamp']
            if current_step == 0:  # 最初のタイムステップでショート
                size = 0.1
                return 'sell', size
            elif current_step == 1:  # 次のタイムステップで買い戻す
                return 'close_short', positions[0]['size']
            return 'hold', 0
    # ショート取引用のデータと戦略
    test_data = generate_short_test_data()
    strategy = ShortTestStrategy()

    # 初期化
    engine = SimulationEngine(
        initial_balance=10000,  # 初期資産
        price_data=test_data,
        fee_function=fixed_fee,  # 手数料関数
        output_dir="runs/test_simulation_01"  # 出力を不要にする
    )

    # シミュレーション実行
    final_balance, history, profit_history = engine.run_simulation(strategy, lookback_period=1)

    # 手計算で期待される最終資産を確認
    # 初期価格で10000 * 0.1ショート (10000 * 0.1 = 1000)
    # 次の価格で買い戻し (10000 - 9900) * 0.1 = 10
    # 手数料 (10000 + 9900) * 0.01 = 199
    expected_profit = (10000 - 9900) * 0.1 - fixed_fee(10000 * 0.1) - fixed_fee(9900 * 0.1)
    expected_balance = 10000 + expected_profit

    # アサーション
    assert abs(final_balance['cash'] - expected_balance) < 1e-6, f"Test failed: {final_balance['cash']} != {expected_balance}"
    print(f"Test passed: Final balance {final_balance['cash']} matches expected balance {expected_balance}")

