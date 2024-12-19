import os
import numpy as np
import pandas as pd
import time
import csv
from typing import Callable, List, Dict, Tuple, Optional

class SimulationEngine:
    """
    A simulation engine for backtesting trading strategies on historical price data.

    Attributes:
        balance (Dict[str, float]): Current balances of cash.
        positions (List[Dict]): Active positions (long and short).
        price_data (pd.DataFrame): Historical price data for the simulation.
        history (List[Dict]): Trade execution history.
        profit_history (List[Dict]): Profit history over time.
        total_fees (float): Total fees incurred during the simulation.
        output_dir (str): Directory to save trade history and profit history.
        fee_function (Callable[[float], float]): Function to calculate transaction fees.
        market_data_function (Callable): Function to retrieve market data.
    """
    def __init__(
        self, 
        initial_balance: float, 
        price_data: pd.DataFrame, 
        fee_function: Callable[[float], float], 
        output_dir: str = 'runs',
        market_data_function: Optional[Callable[[int, int], pd.DataFrame]] = None
    ) -> None:
        self.balance: Dict[str, float] = {'cash': initial_balance}
        self.positions: List[Dict] = []  # Multiple position support
        self.price_data: pd.DataFrame = price_data
        self.history: List[Dict] = []  # Stores trade history
        self.profit_history: List[Dict] = []  # Tracks profit over time
        self.total_fees: float = 0
        self.save_path: str = output_dir
        self.fee_function: Callable[[float], float] = fee_function
        self.market_data_function: Callable[[int, int], pd.DataFrame] = market_data_function or self.default_market_data_function

    def generate_metrics(self) -> None:
        """
        Generate and display trading metrics, such as:
        - Total Profit/Loss
        - Maximum Drawdown
        - Sharpe Ratio
        """
        # Profit History as a Series
        profit_series = pd.Series([record['profit'] for record in self.profit_history])

        # Total Profit/Loss
        total_profit = profit_series.iloc[-1] if not profit_series.empty else 0

        # Max Drawdown Calculation
        cumulative_max = profit_series.cummax()
        drawdown = profit_series - cumulative_max
        max_drawdown = drawdown.min()

        # Sharpe Ratio Calculation
        returns = profit_series.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if not returns.empty else 0

        # Display Metrics
        print("\n=== Simulation Metrics ===")
        print(f"Total Profit/Loss: {total_profit:.2f} USD")
        print(f"Maximum Drawdown: {max_drawdown:.2f} USD")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print("==========================\n")

    def default_market_data_function(self, timestamp: int, lookback_period: int) -> pd.DataFrame:
        """Fetches historical market data based on the timestamp and lookback period."""
        index = self.price_data.index[self.price_data['timestamp'] == timestamp][0]
        start_index = max(0, index - lookback_period)
        if start_index == index:  # Return empty DataFrame if insufficient data
            return pd.DataFrame(columns=self.price_data.columns)
        return self.price_data.iloc[start_index:index]

    def get_market_data(self, timestamp: int, lookback_period: int) -> pd.DataFrame:
        """Retrieves market data using the market_data_function."""
        return self.market_data_function(timestamp, lookback_period)

    def execute_buy(self, price: float, size: float) -> None:
        """Executes a buy order and updates the balance and positions."""
        cost = price * size
        fee = self.fee_function(cost)
        if self.balance['cash'] >= cost + fee:
            self.balance['cash'] -= cost + fee
            self.total_fees += fee
            # Check or update long position
            existing_position = next((pos for pos in self.positions if pos['type'] == 'long'), None)
            if existing_position:
                # Calculate average entry price
                total_contract_value = (existing_position['size'] * existing_position['entry_price']) + (size * price)
                total_size = existing_position['size'] + size
                new_entry_price = total_contract_value / total_size

                # Update position
                existing_position['size'] = total_size
                existing_position['entry_price'] = new_entry_price
            else:
                self.positions.append({'type': 'long', 'size': size, 'entry_price': price})

            self.history.append({'action': 'buy', 'price': price, 'size': size, 'fee': fee})

        else:
            print("Not enough cash to buy trade balance:{} < cost:{} + fee:{}".format(self.balance['cash'], cost, fee))

    def execute_sell(self, price: float, size: float) -> None:
        """Executes a sell order and updates the balance and positions."""
        cost = price * size
        fee = self.fee_function(cost)

        # Check or update short position
        existing_position = next((pos for pos in self.positions if pos['type'] == 'short'), None)
        if existing_position:
            total_contract_value = (existing_position['size'] * existing_position['entry_price']) + (size * price)
            total_size = existing_position['size'] + size
            new_entry_price = total_contract_value / total_size

            # Update position
            existing_position['size'] = total_size
            existing_position['entry_price'] = new_entry_price
        else:
            self.positions.append({'type': 'short', 'size': size, 'entry_price': price})

        self.balance['cash'] -= fee
        self.total_fees += fee
        self.history.append({'action': 'sell', 'price': price, 'size': size, 'fee': fee})

    def close_position(self, price: float, position_type: str) -> None:
        """Closes a position and updates the balance."""
        position = next((pos for pos in self.positions if pos['type'] == position_type), None)
        if position:
            size = position['size']
            entry_price = position['entry_price']

            if position_type == 'long':
                # ロングポジションの利益計算
                revenue = (price - entry_price) * size
                return_entry_price = entry_price * size
            elif position_type == 'short':
                # ショートポジションの利益計算
                revenue = (entry_price - price) * size
                return_entry_price = 0
            else:
                raise ValueError("Invalid position type. Must be 'long' or 'short'.")
            
            fee = self.fee_function(price * size)
            net_revenue = revenue - fee
            # 現金残高を更新
            self.balance['cash'] += return_entry_price + net_revenue
            self.total_fees += fee

            self.history.append({'action': f'close_{position_type}', 'price': price, 'size': position['size'], 'fee': fee})
            self.positions.remove(position)

    def run_simulation(self, strategy: Callable[[pd.DataFrame, Dict[str, float], List[Dict]], Tuple[str, float]], lookback_period: int = 10) -> [Dict[str, float],List[Dict],List[Dict]]:
        """Runs the simulation using the given strategy."""
        total_steps = len(self.price_data)
        start_time = time.time()

        for i, row in self.price_data.iterrows():
            timestamp = row['timestamp']
            market_data = self.get_market_data(timestamp, lookback_period)

            if market_data.empty:  # Skip if data is insufficient
                continue

            # Execute logic
            action, size = strategy(market_data, self.balance, self.positions)

            before_act_balance = self.balance['cash']

            # Execute trade based on action
            if action == 'buy':
                self.execute_buy(market_data['close'].iloc[-1], size)
                profit = 0
            elif action == 'sell':
                self.execute_sell(market_data['close'].iloc[-1], size)
                profit = 0
            elif action == 'close_long':
                self.close_position(market_data['close'].iloc[-1], 'long')
                profit = self.balance['cash'] - before_act_balance
            elif action == 'close_short':
                self.close_position(market_data['close'].iloc[-1], 'short')
                profit = self.balance['cash'] - before_act_balance
            else:
                profit = 0
            # Calculate and save profit
            self.profit_history.append({'timestamp': timestamp, 'profit': profit})

            # Display progress
            if i % 100 == 0 or i == total_steps - 1:
                elapsed = time.time() - start_time
                progress = (i + 1) / total_steps * 100
                print(f"Progress: {progress:.2f}% | Time Elapsed: {elapsed:.2f}s | Profit: {profit:.2f} USD")

        # Generate metrics
        self.generate_metrics()
        # Save results to CSV
        self.save_results()

        return self.balance, self.history, self.profit_history

    def save_results(self) -> None:
        """Saves trade history and profit history to CSV files."""
        print(self.save_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(self.save_path, 'trade_history.csv'), index=False)

        profit_df = pd.DataFrame(self.profit_history)
        profit_df.to_csv(os.path.join(self.save_path, 'profit_history.csv'), index=False)


# 手数料関数（例: 固定割合の手数料 0.1%）
def fixed_fee(amount):
    return amount * 0.01

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


def test_similation():
    # テスト用データ
    price_data = pd.DataFrame({
        'timestamp': range(1000),
        'date': pd.date_range(start='2024-01-01', periods=1000, freq='T'),
        'open': pd.Series(range(1000)) * 1.02,
        'high': pd.Series(range(1000)) * 1.03,
        'low': pd.Series(range(1000)) * 0.97,
        'close': pd.Series(range(1000)) * 1.01,
        'volume': pd.Series(range(1000)) * 10
    })

    # シミュレーションの実行
    engine = SimulationEngine(
        initial_balance=10000,
        price_data=price_data,
        fee_function=fixed_fee
    )

    dummy_strategy = DummyStrategy()
    engine.run_simulation(strategy=dummy_strategy, lookback_period=10)

if __name__ == "__main__":
    test_similation()