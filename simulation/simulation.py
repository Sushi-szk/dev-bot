import os
import numpy as np
import pandas as pd
import time
import csv
from typing import Callable, List, Dict, Tuple, Optional

class SimulationEngine:
    """
    A simulation engine for backtesting trading strategies on historical price data.
    Supports:
      - Market Orders / Limit Orders
      - Take Profit / Stop Loss
      - Canceling open orders
      - Multiple actions (buy/sell/close) in the same step
    """

    def __init__(
        self, 
        initial_balance: float, 
        price_data: pd.DataFrame, 
        fee_function: Callable[[float], float], 
        output_dir: str = 'runs',
        market_data_function: Optional[Callable[[int, int], pd.DataFrame]] = None
    ) -> None:
        self.initial_balance = initial_balance  # 初期残高
        self.balance: Dict[str, float] = {'cash': initial_balance}  # 口座残高
        self.positions: List[Dict] = []  # ポジションリスト
        self.price_data: pd.DataFrame = price_data  # 全価格データ
        self.market_data: pd.DataFrame = None       # 現在のマーケットデータ(各ステップ)
        self.history: List[Dict] = []               # 取引履歴
        self.profit_history: List[Dict] = []        # 累積純損益(手数料込み)推移履歴
        self.total_fees: float = 0                  # 総手数料
        self.save_path: str = output_dir            # 保存先ディレクトリ
        self.fee_function: Callable[[float], float] = fee_function
        self.market_data_function: Callable[[int, int], pd.DataFrame] = market_data_function or self.default_market_data_function

        # 未約定の指値注文を管理するリスト
        self.open_orders: List[Dict] = []
        self.next_order_id: int = 1

    # ------------------
    # generate_metrics
    # ------------------
    def generate_metrics(self) -> None:
        """
        Calculate final net profit, max drawdown, Sharpe Ratio, etc.
        profit_history: Each step's "cumulative net profit"
        """
        profit_series = pd.Series([record['profit'] for record in self.profit_history])
        if profit_series.empty:
            print("\nNo trades were made; no metrics to display.")
            return

        final_net_profit = profit_series.iloc[-1]

        # ドローダウン
        cummax_ = profit_series.cummax()
        drawdown_ = profit_series - cummax_
        max_drawdown = drawdown_.min()

        # シャープレシオ
        returns = profit_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        
        sharpe_ratio = returns.mean() / returns.std() if not returns.empty and returns.std() != 0 else 0

        print("\n=== Simulation Metrics ===")
        print(f"Final Net Profit:  {final_net_profit:.2f} USD")
        print(f"Total Fees Paid:   {self.total_fees:.2f} USD")
        print(f"Maximum Drawdown:  {max_drawdown:.2f} USD")
        print(f"Sharpe Ratio:      {sharpe_ratio:.2f}")
        print("==========================\n")

    def default_market_data_function(self, timestamp: int, lookback_period: int) -> pd.DataFrame:
        """Fetches historical market data based on the timestamp and lookback period."""
        index_list = self.price_data.index[self.price_data['timestamp'] == timestamp].tolist()
        if not index_list:
            return pd.DataFrame(columns=self.price_data.columns)
        index = index_list[0]
        start_index = max(0, index - lookback_period)
        if start_index == index:
            return pd.DataFrame(columns=self.price_data.columns)
        return self.price_data.iloc[start_index:index]

    def get_market_data(self, timestamp: int, lookback_period: int) -> pd.DataFrame:
        """Retrieves market data using the market_data_function."""
        return self.market_data_function(timestamp, lookback_period)

    # ----------------------------------------------------------------
    # 注文関連メソッド (Market / Limit)
    # ----------------------------------------------------------------
    def submit_order(
        self, 
        side: str, 
        size: float, 
        limit_price: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None
    ) -> None:
        """
        side: 'buy' or 'sell'
        size: 注文サイズ
        limit_price: 指値価格 (Noneなら成り行き)
        take_profit: 利確価格
        stop_loss: 損切価格
        """
        if size <= 0:
            return
        if limit_price is not None and limit_price > 0:
            order_id = self.next_order_id
            self.next_order_id += 1
            self.open_orders.append({
                'id': order_id,
                'side': side,
                'size': size,
                'limit_price': limit_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss
            })
        else:
            # Market Order
            current_price = self.market_data['close'].iloc[-1]
            self._execute_market_order(side, current_price, size, take_profit, stop_loss)


    def _execute_market_order(
        self, 
        side: str, 
        price: float, 
        size: float, 
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None
    ) -> None:
        """内部呼び出し: 約定価格が確定している(成り行き or 指値約定)"""
        if side == 'buy':
            self.execute_buy(price, size, take_profit, stop_loss)
        elif side == 'sell':
            self.execute_sell(price, size, take_profit, stop_loss)

    def cancel_order(self, order_id: int) -> None:
        """指定オーダーIDをキャンセル(= open_ordersから削除)"""
        self.open_orders = [o for o in self.open_orders if o['id'] != order_id]

    def process_open_orders(self, current_row: pd.Series) -> None:
        """
        指値注文の約定可否をチェックして約定実行。
        current_row はローソク足(high, low, ... など)
        """
        if not self.open_orders:
            return

        filled_orders = []
        current_high = current_row['high']
        current_low = current_row['low']

        for order in self.open_orders:
            side = order['side']
            limit_price = order['limit_price']
            size = order['size']
            tp = order.get('take_profit')
            sl = order.get('stop_loss')

            if side == 'buy':
                # buy limit => low <= limit_price で約定
                if current_low <= limit_price:
                    self._execute_market_order('buy', limit_price, size, tp, sl)
                    filled_orders.append(order)
            elif side == 'sell':
                # sell limit => high >= limit_price で約定
                if current_high >= limit_price:
                    self._execute_market_order('sell', limit_price, size, tp, sl)
                    filled_orders.append(order)

        # 約定済み注文を open_orders から消す
        for fo in filled_orders:
            self.open_orders.remove(fo)

    # ----------------------------------------------------------------
    # ポジション管理
    # ----------------------------------------------------------------
    def execute_buy(self, price: float, size: float,
                    take_profit: Optional[float] = None,
                    stop_loss: Optional[float] = None
    ) -> None:
        cost = price * size
        entry_fee = self.fee_function(cost)

        # 口座残高から購入額(デリバティブであれば必要証拠金)を引く
        total_spend = cost + entry_fee
        if self.balance['cash'] < total_spend:
            print(f"Not enough cash to buy. Need={total_spend:.2f}, bal={self.balance['cash']:.2f}")
            return

        self.balance['cash'] -= total_spend
        self.total_fees += entry_fee

        # 既存ロングに追加 or 新規ロング
        existing_long = next((p for p in self.positions if p['type'] == 'long'), None)
        if existing_long:
            # 平均取得単価更新
            total_val = (existing_long['size'] * existing_long['entry_price']) + (size * price)
            total_size = existing_long['size'] + size
            new_entry_price = total_val / total_size

            # entry_fee の扱い: 既存ポジションにまとめるか、個別管理するか悩ましいところ
            # ここでは単純に "追加入庫" した分だけ entry_fee を足す形にする (累積)
            existing_long['size'] = total_size
            existing_long['entry_price'] = new_entry_price
            existing_long['entry_fee'] += entry_fee  # 追加入庫分の手数料を足す

            if take_profit is not None:
                existing_long['take_profit'] = take_profit
            if stop_loss is not None:
                existing_long['stop_loss'] = stop_loss
        else:
            # 新規ロング
            pos = {
                'type': 'long',
                'size': size,
                'entry_price': price,
                'entry_fee': entry_fee,  # エントリー手数料を記録
                'take_profit': take_profit,
                'stop_loss': stop_loss
            }
            self.positions.append(pos)

        # 取引履歴
        self.history.append({
            'action': 'buy',
            'price': price,
            'size': size,
            'fee': entry_fee,
            'info': 'No realized PnL at entry (stored fee internally).'
        })


    def execute_sell(self, price: float, size: float,
                     take_profit: Optional[float] = None,
                     stop_loss: Optional[float] = None
    ) -> None:
        cost = price * size
        entry_fee = self.fee_function(cost)

        # 口座残高から必要証拠金を引く
        total_spend = cost + entry_fee
        if self.balance['cash'] < total_spend:
            print(f"Not enough cash for short fee. fee={entry_fee:.2f}, bal={self.balance['cash']:.2f}")
            return

        self.balance['cash'] -= total_spend
        self.total_fees += entry_fee

        existing_short = next((p for p in self.positions if p['type'] == 'short'), None)
        if existing_short:
            total_val = (existing_short['size'] * existing_short['entry_price']) + (size * price)
            total_size = existing_short['size'] + size
            new_entry_price = total_val / total_size
            existing_short['size'] = total_size
            existing_short['entry_price'] = new_entry_price
            existing_short['entry_fee'] += entry_fee

            if take_profit is not None:
                existing_short['take_profit'] = take_profit
            if stop_loss is not None:
                existing_short['stop_loss'] = stop_loss
        else:
            pos = {
                'type': 'short',
                'size': size,
                'entry_price': price,
                'entry_fee': entry_fee,
                'take_profit': take_profit,
                'stop_loss': stop_loss
            }
            self.positions.append(pos)

        self.history.append({
            'action': 'sell',
            'price': price,
            'size': size,
            'fee': entry_fee,
            'info': 'No realized PnL at entry (stored fee internally).'
        })

    # ----------------------------------------------------------------
    # ポジションクローズ
    # ----------------------------------------------------------------
    def close_position(self, pos_idx: int, close_price: float, close_size: float = float('nan')) -> float:
        """
        pos_idx 番目のポジションを、指定サイズだけクローズする(部分 or 全部)。
        戻り値 => 今回クローズされたサイズに対応する実現損益(手数料控除後)。
        """

        position = self.positions[pos_idx]
        pos_type = position['type']
        pos_size = position['size']
        entry_price = position['entry_price']
        entry_fee = position['entry_fee']  # これまでに累積されているエントリー手数料

        # クローズサイズのバリデーション
        if close_size <= 0:
            print("Close size must be > 0. Doing nothing.")
            return 0.0
        if close_size > pos_size or close_size is float('nan'):
            print(f"Close size {close_size} exceeds or is None position size {pos_size}. Will close entire position instead.")
            close_size = pos_size

        # グロス損益 (ロングの場合: (決済価格-建値)*サイズ, ショートの場合は逆)
        if pos_type == 'long':
            gross_pnl = (close_price - entry_price) * close_size
        elif pos_type == 'short':
            gross_pnl = (entry_price - close_price) * close_size
        else:
            raise ValueError("Invalid position type.")

        # クローズ時の手数料 (約定代金= close_price * close_size に対して計算)
        close_fee = self.fee_function(close_price * close_size)
        self.total_fees += close_fee

        # エントリー手数料のうち、今回クローズするサイズに対応する分だけを按分して差し引く
        # 例: ポジション3BTCのうち1BTCをクローズ => entry_feeの1/3を引く
        partial_entry_fee = entry_fee * (close_size / pos_size)

        # net pnl
        net_pnl = gross_pnl - partial_entry_fee - close_fee

        # 残高を更新
        #  - "entry_price * close_size" は「建玉に紐づく原資(ロングの場合は買いコスト / ショートの場合は担保など)を戻す」イメージ
        #  - "partial_entry_fee" は、エントリー時に先払いしていた手数料分の一部を戻してから差し引く形をとる
        #    ただし上式で net_pnl に既に `- partial_entry_fee` しているので、最終的には net_pnl 分だけ資金が増減する。
        #  ここでは、これまでに行っていたロジックに合わせて
        #  net_pnl + partial_entry_fee + (entry_price * close_size) を加算する形を維持。
        self.balance['cash'] += (net_pnl + partial_entry_fee) + (entry_price * close_size)

        # history に記録
        self.history.append({
            'action': f'close_{pos_type}',
            'price': close_price,
            'size': close_size,          # 部分クローズしたサイズを明示
            'gross_pnl': gross_pnl,
            'entry_fee': partial_entry_fee,
            'close_fee': close_fee,
            'net_pnl': net_pnl
        })

        # ポジションをアップデート(部分 or 全部クローズ)
        remaining_size = pos_size - close_size

        if remaining_size > 0:
            # 部分クローズなので、サイズと entry_fee を引き下げる
            position['size'] = remaining_size
            position['entry_fee'] -= partial_entry_fee
            # 建値に関しては、そのままにするか、再計算するかは要件次第
            # position['entry_price'] = ... 
            # ここでは据え置きにする(実装ポリシーによっては要変更)
        else:
            # 全部クローズ
            del self.positions[pos_idx]

        return net_pnl


    # ----------------------------------------------------------------
    # 利確・損切 (TP/SL) チェック
    # ----------------------------------------------------------------
    def check_positions_tp_sl(self, current_row: pd.Series) -> float:
        """
        各ポジションの TP/SL をチェックし、該当すれば close_position を呼ぶ。
        同時到達なら SL 優先でクローズ。

        戻り値 => このステップで発生した実現損益の合計
        """
        if not self.positions:
            return 0.0

        total_closed_pnl = 0.0
        high_ = current_row['high']
        low_ = current_row['low']

        for i in reversed(range(len(self.positions))):
            pos = self.positions[i]
            tp = pos.get('take_profit')
            sl = pos.get('stop_loss')
            if tp is None and sl is None:
                continue

            pos_type = pos['type']
            triggered_sl = False
            triggered_tp = False

            if pos_type == 'long':
                if sl is not None and low_ <= sl:
                    triggered_sl = True
                if tp is not None and high_ >= tp:
                    triggered_tp = True

                # 同時トリガーならSL優先
                if triggered_sl and triggered_tp:
                    total_closed_pnl += self.close_position(i, sl)
                elif triggered_sl:
                    total_closed_pnl += self.close_position(i, sl)
                elif triggered_tp:
                    total_closed_pnl += self.close_position(i, tp)

            elif pos_type == 'short':
                if sl is not None and high_ >= sl:
                    triggered_sl = True
                if tp is not None and low_ <= tp:
                    triggered_tp = True

                if triggered_sl and triggered_tp:
                    total_closed_pnl += self.close_position(i, sl)
                elif triggered_sl:
                    total_closed_pnl += self.close_position(i, sl)
                elif triggered_tp:
                    total_closed_pnl += self.close_position(i, tp)

        return total_closed_pnl

    # -----------------------------
    # run_simulation
    # -----------------------------
    def run_simulation(
        self, 
        strategy, 
        lookback_period: int = 10
    ) -> Tuple[Dict[str, float], List[Dict], List[Dict]]:
        """
        strategy: 
          - cancel_orders(open_orders, market_data, balance, positions) -> List[int]
          - __call__(market_data, balance, positions) -> List of (action, size, limit_price, tp, sl)
        """
        total_steps = len(self.price_data)
        start_time = time.time()

        # 累計の実現損益
        cumulative_pnl = 0.0

        for i, row in self.price_data.iterrows():
            timestamp = row['timestamp']
            self.market_data = self.get_market_data(timestamp, lookback_period)
            if self.market_data.empty:
                continue

            # 1) 指値注文の約定チェック
            self.process_open_orders(row)

            # 2) TP/SL チェック => net_pnl が返ってくる
            step_closed_pnl = self.check_positions_tp_sl(row)

            # 3) Strategyでキャンセル
            cancel_ids = strategy.cancel_orders(self.open_orders, self.market_data, self.balance, self.positions)
            for cid in cancel_ids:
                self.cancel_order(cid)

            # 4) Strategyアクション(複数)
            actions = strategy(self.market_data, self.balance, self.positions)
            if not isinstance(actions, list):
                actions = [actions]

            step_manual_close_pnl = 0.0  # 追加で close_long/close_short した場合のPnL
            for (action, size, limit_price, tp, sl) in actions:
                if action == 'buy':
                    self.submit_order('buy', size, limit_price, tp, sl)
                elif action == 'sell':
                    self.submit_order('sell', size, limit_price, tp, sl)
                elif action == 'close_long':
                    idx = next((ix for ix, p in enumerate(self.positions) if p['type'] == 'long'), None)
                    if idx is not None:
                        current_price = self.market_data['close'].iloc[-1]
                        step_manual_close_pnl += self.close_position(idx, current_price, size)
                elif action == 'close_short':
                    idx = next((ix for ix, p in enumerate(self.positions) if p['type'] == 'short'), None)
                    if idx is not None:
                        current_price = self.market_data['close'].iloc[-1]
                        step_manual_close_pnl += self.close_position(idx, current_price,size)
                elif action == 'hold':
                    pass
                else:
                    print(f"Unknown action: {action}")

            # このステップで発生した実現PnL(クローズによる)
            step_pnl = step_closed_pnl + step_manual_close_pnl

            # 累計に加算
            cumulative_pnl += step_pnl

            # profit_history に記録 (timestamp + 現在の累計PnL)
            self.profit_history.append({
                'timestamp': timestamp,
                'profit': cumulative_pnl
            })

            # Display progress
            if i % 100 == 0 or i == total_steps - 1:
                elapsed = time.time() - start_time
                progress = (i + 1) / total_steps * 100
                print(f"Progress: {progress:.2f}% | Elapsed: {elapsed:.2f}s | StepPnL: {step_pnl:.2f} USD")

        # シミュレーション終了
        self.generate_metrics()
        self.save_results()
        return self.balance, self.history, self.profit_history

    def save_results(self) -> None:
        """Saves trade history and profit history to CSV."""
        print(self.save_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        pd.DataFrame(self.history).to_csv(os.path.join(self.save_path, 'trade_history.csv'), index=False)
        pd.DataFrame(self.profit_history).to_csv(os.path.join(self.save_path, 'profit_history.csv'), index=False)


# 手数料関数の例(固定割合1%)
def fixed_fee(amount):
    return amount * 0.01


class MultiActionStrategy:
    """
    デモ用ストラテジ:
    1. 最初の3ステップで毎ステップ「buy(成り行き)」アクションを追加
    2. 同じステップでショートを出してみたり、ロングをクローズしたりする例を混ぜる

    戻り値の形式: List[Tuple[str, float, Optional[float], Optional[float], Optional[float]]]
    例: [('buy', 0.1, None, None, None), ('close_short', 0.0, None, None, None)]
    """

    def cancel_orders(
        self, 
        open_orders: List[Dict], 
        market_data: pd.DataFrame, 
        balance: Dict[str, float], 
        positions: List[Dict]
    ) -> List[int]:
        # ここではキャンセルしない例
        return []

    def __call__(
        self, 
        market_data: pd.DataFrame, 
        balance: Dict[str, float], 
        positions: List[Dict]
    ) -> List[Tuple[str, float, Optional[float], Optional[float], Optional[float]]]:
        if market_data.empty:
            return []

        step = market_data.iloc[-1]['timestamp']
        current_price = market_data['close'].iloc[-1]

        actions = []
        # 例: step < 3 => 毎ステップ0.1BTCを成り行きでbuy
        if step < 3:
            actions.append(('buy', 0.1, None, None, None))

        # 例: step == 2 => 同時にショート注文(指値)も出す(わざと両建てを発生させる)
        if step == 2:
            limit_price = current_price * 1.01
            actions.append(('sell', 0.1, limit_price, None, None))

        # 例: step == 4 => ロングを全クローズ、かつ同時にショートを成り行きで建てる
        # (あまり意味のないトレードだがマルチアクションの例として)
        if step == 4:
            actions.append(('close_long', 0.0, None, None, None))
            actions.append(('sell', 0.1, None, None, None))

        # 例: step == 5 => ショートをクローズ
        if step == 5:
            actions.append(('close_short', 0.0, None, None, None))

        return actions


def test_simulation():
    """簡単なテスト実行"""
    price_data = pd.DataFrame({
        'timestamp': range(10),
        'date': pd.date_range(start='2024-01-01', periods=10, freq='min'),
        'open':  [10000+i*10 for i in range(10)],
        'high':  [10000+i*10+5 for i in range(10)],
        'low':   [10000+i*10-5 for i in range(10)],
        'close': [10000+i*10 for i in range(10)],
        'volume': [100+i for i in range(10)]
    })

    engine = SimulationEngine(
        initial_balance=10000,
        price_data=price_data,
        fee_function=fixed_fee,
        output_dir="runs/multiaction_demo"
    )

    strategy = MultiActionStrategy()
    final_balance, history, profit_history = engine.run_simulation(strategy=strategy, lookback_period=1)

    print("Final balance:", final_balance)
    print("History:", history)
    print("Profit History:", profit_history)

if __name__ == "__main__":
    test_simulation()
