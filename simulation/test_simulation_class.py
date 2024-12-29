from simulation import SimulationEngine

import pandas as pd
from typing import List, Dict, Tuple, Optional

def fixed_fee(amount):
    return amount * 0.01

# SimulationEngineのテスト
def test_simulation_engine_simple_long():
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
        def cancel_orders(
            self, 
            open_orders: List[Dict], 
            market_data: pd.DataFrame, 
            balance: Dict[str, float], 
            positions: List[Dict]
        ) -> List[int]:
            """
            - 「現在の価格やポジション状況を見て、不要な注文はキャンセルする」ロジックを書く
            - 例: 現在価格より5%以上離れている注文はキャンセルするとか、時間が経過したらキャンセルとか
            - ここではデモとして、単純にすべての open_orders をキャンセルするサンプル実装
            """
            #return [order['id'] for order in open_orders]  # すべてキャンセル
            return []

        def __call__(self, market_data: pd.DataFrame, balance: Dict[str, float], positions: List[Dict]) -> Tuple[str, float]:
            # 現在のタイムステップを取得
            current_step = market_data.iloc[-1]['timestamp']
            if current_step == 1:  # 2番目のタイムステップで買う
                size = 0.1
                return 'buy', size, None, None, None
            elif current_step == 2:  # 次のタイムステップで売る
                return 'close_long', positions[0]['size'], None, None, None
            return 'hold', 0, None, None, None
            
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



def test_simulation_engine_simple_short():
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
        def cancel_orders(
            self, 
            open_orders: List[Dict], 
            market_data: pd.DataFrame, 
            balance: Dict[str, float], 
            positions: List[Dict]
        ) -> List[int]:
            """
            - 「現在の価格やポジション状況を見て、不要な注文はキャンセルする」ロジックを書く
            - 例: 現在価格より5%以上離れている注文はキャンセルするとか、時間が経過したらキャンセルとか
            - ここではデモとして、単純にすべての open_orders をキャンセルするサンプル実装
            """
            #return [order['id'] for order in open_orders]  # すべてキャンセル
            return []

        def __call__(self, market_data: pd.DataFrame, balance: Dict[str, float], positions: List[Dict]) -> Tuple[str, float]:
            # 現在のタイムステップを取得
            current_step = market_data.iloc[-1]['timestamp']
            if current_step == 0:  # 最初のタイムステップでショート
                size = 0.1
                return 'sell', size, None, None, None
            elif current_step == 1:  # 次のタイムステップで買い戻す
                return 'close_short', positions[0]['size'], None, None, None
            return 'hold', 0, None, None, None

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

    print(history)
    # 手計算で期待される最終資産を確認
    # 初期価格で10000 * 0.1ショート (10000 * 0.1 = 1000)
    # 次の価格で買い戻し (10000 - 9900) * 0.1 = 10
    # 手数料 (10000 + 9900) * 0.01 = 199
    expected_profit = (10000 - 9900) * 0.1 - fixed_fee(10000 * 0.1) - fixed_fee(9900 * 0.1)
    expected_balance = 10000 + expected_profit

    # アサーション
    assert abs(final_balance['cash'] - expected_balance) < 1e-6, f"Test failed: {final_balance['cash']} != {expected_balance}"
    print(f"Test passed: Final balance {final_balance['cash']} matches expected balance {expected_balance}")

#
# テスト追加例1: 指値注文が約定するか確認
#
def test_simulation_engine_limit_order_fill():
    """
    指値BUY注文を出し、次の足でその価格以下になったら約定し、
    その後に決済して終わるか確認するテスト。
    """

    # シンプルに価格が少しずつ下がるようにデータを作る
    data = pd.DataFrame({
        'timestamp': range(5),
        'date': pd.date_range(start='2024-01-01', periods=5, freq='min'),
        'open':  [10000, 9990,  9980,  9970,  9960],
        'high':  [10010, 10000, 9990,  9980,  9970],
        'low':   [9990,  9980,  9970,  9960,  9950],
        'close': [9995,  9985,  9975,  9965,  9955],
    })

    class LimitBuyStrategy:
        """
        1. 最初に指値BUY注文を出す (価格は9990)
        2. 指値約定後、即 close_long する
        """
        def __init__(self):
            self.order_submitted = False
            self.order_filled = False

        def cancel_orders(
            self, 
            open_orders: List[Dict], 
            market_data: pd.DataFrame, 
            balance: Dict[str, float], 
            positions: List[Dict]
        ) -> List[int]:
            # 今回はキャンセルしない
            return []

        def __call__(
            self,
            market_data: pd.DataFrame, 
            balance: Dict[str, float], 
            positions: List[Dict]
        ) -> Tuple[str, float, Optional[float], Optional[float], Optional[float]]:
            current_step = market_data.iloc[-1]['timestamp']
            current_price = market_data.iloc[-1]['close']
            
            # まだ注文を出していなければ、最初のタイミング(0番目)で指値買い注文を出す
            if current_step == 0 and not self.order_submitted:
                self.order_submitted = True
                return ('buy', 0.1, 9990, None, None)  # 指値9990で買い

            # ポジションが約定して持てていれば、クローズ(成り行き)
            long_pos = next((p for p in positions if p['type'] == 'long'), None)
            if long_pos and not self.order_filled:
                # 1回でもポジションを持てたら "order_filled" とみなす
                self.order_filled = True
                return ('close_long', long_pos['size'], None, None, None)

            return ('hold', 0, None, None, None)

    engine = SimulationEngine(
        initial_balance=10000,
        price_data=data,
        fee_function=fixed_fee,
        output_dir="runs/test_simulation_limit_order_fill"
    )
    strategy = LimitBuyStrategy()

    final_balance, history, profit_history = engine.run_simulation(strategy=strategy, lookback_period=1)

    # テスト
    # 1) 指値買いが約定しているはず
    # 2) その後 close_long が実行されているはず

    # 注文履歴に "buy" / "close_long" が含まれるかどうか
    actions = [h['action'] for h in history]
    assert 'buy' in actions, "Limit buy order must be filled at some point."
    assert any(a.startswith('close_long') for a in actions), "After fill, it must close the long position."
    
    # ポジションが最終的にゼロになっている
    assert len(engine.positions) == 0, "All positions should be closed at the end."

    # ざっくり残高が初期値より減っていないかチェック(手数料だけ差し引かれる想定)
    # ここでは大まかなチェックに留めるが、必要なら exact な数値も計算可能
    assert final_balance['cash'] < 10000, "We must pay some fees, so final balance < initial balance"
    print("test_simulation_engine_limit_order_fill passed.")


#
# テスト追加例2: 指値注文を出したあとでStrategyがキャンセルし、約定しないことを確認
#
def test_simulation_engine_limit_order_cancel():
    """
    指値買い注文を出すが、価格に到達する前にキャンセルするテスト。
    その後ポジションは持たないまま終了するはず。
    """

    data = pd.DataFrame({
        'timestamp': range(5),
        'date': pd.date_range(start='2024-01-01', periods=5, freq='min'),
        'open':  [10000, 10010, 10020, 10030, 10040],
        'high':  [10010, 10020, 10030, 10040, 10050],
        'low':   [9990,  10000, 10010, 10020, 10030],
        'close': [10005, 10015, 10025, 10035, 10045],
    })

    class LimitBuyCancelStrategy:
        """
        1. 最初に指値BUY注文を 9990 で出す
        2. しかし、実際には価格が下がらないので約定しない
        3. 2番目のタイムステップでキャンセルする
        """
        def __init__(self):
            self.order_submitted = False

        def cancel_orders(
            self, 
            open_orders: List[Dict], 
            market_data: pd.DataFrame, 
            balance: Dict[str, float], 
            positions: List[Dict]
        ) -> List[int]:
            current_step = market_data.iloc[-1]['timestamp']
            if current_step == 1:
                # 2番目の足で全キャンセル
                return [o['id'] for o in open_orders]
            return []

        def __call__(
            self, 
            market_data: pd.DataFrame, 
            balance: Dict[str, float], 
            positions: List[Dict]
        ) -> Tuple[str, float, Optional[float], Optional[float], Optional[float]]:
            current_step = market_data.iloc[-1]['timestamp']
            
            if current_step == 0 and not self.order_submitted:
                self.order_submitted = True
                return ('buy', 0.1, 9990, None, None)  # 下がらない指値

            # 価格は常に高く、9990 に届かないので約定せず
            return ('hold', 0, None, None, None)

    engine = SimulationEngine(
        initial_balance=10000,
        price_data=data,
        fee_function=fixed_fee,
        output_dir="runs/test_simulation_limit_order_cancel"
    )
    strategy = LimitBuyCancelStrategy()

    final_balance, history, profit_history = engine.run_simulation(strategy=strategy, lookback_period=1)
    
    # 注文履歴に "buy" は存在しない → ここでいう "buy" は約定時ログなので、未約定の場合は記録されない
    # open_orders には、一度指値注文が追加されたがキャンセルされているはず

    # 1) history 内に actual "buy" (約定) が無いかチェック
    assert not any(h['action'] == 'buy' for h in history), "No buy fill should happen"
    # 2) 最終的にポジションはゼロ
    assert len(engine.positions) == 0, "No positions should remain if order was never filled"
    # 3) キャンセル後の最終残高は変化なし(手数料なし)
    assert final_balance['cash'] == 10000, "No fill means no fee deduction"
    print("test_simulation_engine_limit_order_cancel passed.")


#
# テスト追加例3: 複数回BUYしてポジションを積み増しし、最後にまとめてクローズ
#
def test_simulation_engine_multiple_position_long():
    """
    1. 価格がじわじわ上がっていく
    2. 毎足0.1ずつ成り行き買いをしてポジションを積み上げる (3回)
    3. その後、いっぺんに close_long する
    4. 期待する損益計算が合うか確認
    """
    data = pd.DataFrame({
        'timestamp': range(6),
        'date': pd.date_range(start='2024-01-01', periods=6, freq='min'),
        'open':  [10000, 10100, 10200, 10300, 10400, 10500],
        'high':  [10100, 10200, 10300, 10400, 10500, 10600],
        'low':   [9900,  10000, 10100, 10200, 10300, 10400],
        'close': [10000, 10100, 10200, 10300, 10400, 10500],
    })

    class MultiBuyStrategy:
        def __init__(self):
            self.buy_count = 0

        def cancel_orders(self, open_orders, market_data, balance, positions):
            return []

        def __call__(self, market_data, balance, positions):
            step = market_data.iloc[-1]['timestamp']
            # 0~2ステップ目で毎回0.1BTCを成り行きBUY
            # 3ステップ目以降でまとめてクローズ
            if step < 3:
                self.buy_count += 1
                return ('buy', 0.1, None, None, None)
            else:
                long_pos = next((p for p in positions if p['type'] == 'long'), None)
                if long_pos:
                    return ('close_long', long_pos['size'], None, None, None)
                return ('hold', 0, None, None, None)

    engine = SimulationEngine(
        initial_balance=10000,
        price_data=data,
        fee_function=fixed_fee,
        output_dir="runs/test_simulation_engine_multiple_position_long"
    )
    strategy = MultiBuyStrategy()
    final_balance, history, profit_history = engine.run_simulation(strategy=strategy, lookback_period=1)

    # 検証:
    # 1) 'buy' が3回呼ばれているか
    buy_actions = [h for h in history if h['action'] == 'buy']
    assert len(buy_actions) == 3, "We should have 3 buy actions."

    # 2) close_long が1回
    close_long_actions = [h for h in history if h['action'].startswith('close_long')]
    assert len(close_long_actions) == 1, "We should close the position exactly once."

    # 3) 最終的にはポジション0
    assert len(engine.positions) == 0, "No positions left."

    # 4) 利益計算(ざっくりチェック)
    #   - 0ステップ: BUY at 10000(0.1) cost=1000 fee=10
    #   - 1ステップ: BUY at 10100(0.1) cost=1010 fee=10.1
    #   - 2ステップ: BUY at 10200(0.1) cost=1020 fee=10.2
    #
    #   => avg price = (1000+1010+1020)/(0.3) = 1010
    #   => 3ステップ目(10300)でまとめてクローズ(ここでは4ステップ目のcloseが出る前にクローズされる想定)
    #   => revenue = (10300 - 1010*10) * 0.3 ... ん？計算が違うので書き直し
    #
    # 正確に計算する場合、Strategy の動きが「3ステップ目でまとめてクローズ」かどうか要確認：
    #   step=0 => buy( close=10000 )
    #   step=1 => buy( close=10100 )
    #   step=2 => buy( close=10200 )
    #   step=3 => close_long ( close=10300 )
    # 
    # まとめると：
    #   - 1回目 BUY: price=10000, size=0.1, fee=1000*0.01=10
    #   - 2回目 BUY: price=10100, size=0.1, fee=1010*0.01=10.1
    #   - 3回目 BUY: price=10200, size=0.1, fee=1020*0.01=10.2
    # ポジションの平均取得単価:
    #   既存size=0 → 1回目で entry=10000
    #   2回目: total= (0.1*10000 + 0.1*10100) = 1000 + 1010=2010 => size=0.2 => avg=2010/0.2=10050
    #   3回目: total= (0.2*10050 + 0.1*10200) = 2010 + 1020=3030 => size=0.3 => avg=3030/0.3=10100
    #
    # 4回目( step=3 )で close_long: price=10300, size=0.3
    #   revenue = (10300 - 10100)*0.3= (200)*0.3=60
    #   fee = 10300*0.3*0.01= 30.9
    #   net= 60-30.9=29.1
    #
    # したがって：
    #   総手数料= 10 +10.1 +10.2 +30.9=61.2
    #   トレード前残高=10000
    #   最終残高= 10000 - (1000+1010+1020) + (クローズ時返ってくるentry原資) + net
    # 
    # ただしロングなので建玉原資は実際に引かれてる:
    #   1回目BUY: 残高=10000-(1000+10)=8990
    #   2回目BUY: 残高=8990-(1010+10.1)= 7969.9
    #   3回目BUY: 残高=7969.9-(1020+10.2)= 6939.7
    # クローズ: 返却されるのは entry_price(10100)*size(0.3)=3030 + net(29.1)=3059.1
    #  => final balance= 6939.7 +3059.1= 9998.8
    #
    expected_balance_approx = 9998.8

    # 許容誤差内で一致しているか
    assert abs(final_balance['cash'] - expected_balance_approx) < 1e-1, \
        f"Multiple buy final balance mismatch: got {final_balance['cash']}, expected approx {expected_balance_approx}"

    print("test_simulation_engine_multiple_position_long passed.")


#
# テスト追加例4: ロングとショートを同時に建ててみる(ヘッジ)
#
def test_simulation_engine_hedge_positions():
    """
    ロング・ショートを同時に建ててみる。
    単純に2ステップで買い、3ステップで売り、といった形で
    両方ポジションが存在する状態を確認し、最後に同タイミングでクローズする。
    """

    data = pd.DataFrame({
        'timestamp': range(5),
        'date': pd.date_range(start='2024-01-01', periods=5, freq='min'),
        'open':  [10000, 10010, 10020, 10030, 10040],
        'high':  [10010, 10020, 10030, 10040, 10050],
        'low':   [9990,  10000, 10010, 10020, 10030],
        'close': [10000, 10010, 10020, 10030, 10040],
    })

    class HedgeStrategy:
        def cancel_orders(self, open_orders, market_data, balance, positions):
            return []

        def __call__(self, market_data, balance, positions):
            step = market_data.iloc[-1]['timestamp']

            # step=1でロング, step=2でショート
            # step=3で両方クローズ
            if step == 1:
                return ('buy', 0.1, None, None, None)
            elif step == 2:
                return ('sell', 0.1, None, None, None)
            else:
                long_pos = next((p for p in positions if p['type'] == 'long'), None)
                short_pos = next((p for p in positions if p['type'] == 'short'), None)
                if step == 3:
                    # 両方クローズ
                    actions = []
                    actions.append(('close_long', long_pos['size'], None, None, None))
                    actions.append(('close_short', short_pos['size'], None, None, None))
                    return actions

            return ('hold', 0, None, None, None)

    engine = SimulationEngine(
        initial_balance=10000,
        price_data=data,
        fee_function=fixed_fee,
        output_dir="runs/test_simulation_engine_hedge_positions"
    )
    strategy = HedgeStrategy()
    final_balance, history, profit_history = engine.run_simulation(strategy=strategy, lookback_period=1)

    # 検証:
    # step=1 => buy(0.1)
    # step=2 => sell(0.1)
    # step=3 => close_long or close_short (Strategyではlong→shortの順に1回だけ返している。もう1回呼ばれていればshort→longと繰り返すかも)

    # 1) historyに "buy" と "sell" がある
    assert any(h['action'] == 'buy' for h in history), "Must have a buy action"
    assert any(h['action'] == 'sell' for h in history), "Must have a sell action"

    # 2) positions が最終的に空
    assert len(engine.positions) == 0, "Both positions must be closed"

    # 3) 手数料が発生して残高は10000未満のはず(一応チェック)
    assert final_balance['cash'] < 10000, "Fees reduce final balance"

    print("test_simulation_engine_hedge_positions passed.")

def test_simulation_engine_partial_close():
    """
    部分クローズのテスト:
      1) step=0 でロング 0.2BTC
      2) step=1 で 0.1BTC だけ部分クローズ
      3) step=2 で 残り 0.1BTC をすべてクローズ
      価格は単純に step ごとに +100 ずつ上がる想定
      手数料は fixed_fee(amount)=amount*0.01  (1%)
      
      検証観点:
        - 部分クローズ後にポジション size が正しく減っているか
        - 実現損益(net_pnl)を正しく計算できているか
        - 最終残高が期待通りか
    """
    
    def generate_partial_close_test_data() -> pd.DataFrame:
        """シンプルに価格が少しずつ上昇するテストデータを生成。"""
        # 3ステップ + α
        data = pd.DataFrame({
            'timestamp': [0, 1, 2, 3],
            'date': pd.date_range(start='2024-01-01', periods=4, freq='min'),
            'open':  [10000, 10100, 10200, 10300],
            'high':  [10010, 10110, 10210, 10310],
            'low':   [ 9990, 10090, 10190, 10290],
            'close': [10000, 10100, 10200, 10300],
        })
        return data

    class PartialCloseTestStrategy:
        """
        シナリオ:
          - step=0 => buy 0.2 BTC (ロングエントリー)
          - step=1 => close_long 0.1 BTC  (部分クローズ)
          - step=2 => close_long 残り(0.1) 全クローズ
          - 他のステップ => hold
        """
        def cancel_orders(
            self, 
            open_orders: List[Dict], 
            market_data: pd.DataFrame, 
            balance: Dict[str, float], 
            positions: List[Dict]
        ) -> List[int]:
            # 今回はキャンセルしない
            return []

        def __call__(
            self, 
            market_data: pd.DataFrame, 
            balance: Dict[str, float], 
            positions: List[Dict]
        ) -> List[Tuple[str, float, Optional[float], Optional[float], Optional[float]]]:
            step = market_data.iloc[-1]['timestamp']
            actions = []

            if step == 0:
                # 0ステップ目で 0.2BTC ロング
                actions.append(('buy', 0.2, None, None, None))
            elif step == 1:
                # 1ステップ目で 0.1BTC だけ部分クローズ
                # （positions があるかチェックし、ロングがあればそのうちの 0.1）
                long_pos = next((p for p in positions if p['type'] == 'long'), None)
                if long_pos:
                    actions.append(('close_long', 0.1, None, None, None))
            elif step == 2:
                # 残り 0.1BTC 全クローズ
                long_pos = next((p for p in positions if p['type'] == 'long'), None)
                if long_pos:
                    # ここでは 0.0 と指定すると全量クローズの実装にしている場合もありますが、
                    # 今回は size=0.1 を明示して全クローズ
                    actions.append(('close_long', 0.1, None, None, None))
            else:
                # hold
                actions.append(('hold', 0, None, None, None))

            return actions

    # テストデータを生成
    price_data = generate_partial_close_test_data()
    strategy = PartialCloseTestStrategy()

    # エンジン初期化
    engine = SimulationEngine(
        initial_balance=10000.0,
        price_data=price_data,
        fee_function=fixed_fee,
        output_dir="runs/test_partial_close"
    )

    # シミュレーション実行
    final_balance, history, profit_history = engine.run_simulation(strategy=strategy, lookback_period=1)

    print(history)
    # 検証
    #  1) step=0 で buy(0.2BTC, price=10000)
    #    現金= 10000 - 10000*0.2= 10000 - 2000= 8000
    #
    #  2) step=1 で close_long(0.1BTC, price=10100)
    #    gross_pnl= (10100 - 10000)*0.1= 10
    #    partial_entry_fee=  (エントリー手数料20 のうちクローズ比=0.1/0.2=0.5 => 10)
    #    close_fee= (10100*0.1)*0.01= 10.1
    #    net_pnl= 10 - 10 - 10.1= -10.1
    #    => 残高= 8000 + [-10.1 + (10000*0.1)] 
    #       = 8000 + [-10.1 +1000]= 8000 +989.9= 8989.9
    #    ポジション残= 0.1BTC (entry_fee= 10, entry_price=10000 (据え置き))
    #
    #  3) step=2 で close_long(0.1BTC, price=10200)
    #    gross_pnl= (10200 -10000)*0.1= 20
    #    partial_entry_fee= 10*(0.1/0.1)=10
    #    close_fee= (10200*0.1)*0.01= 10.2
    #    net_pnl= 20 -10 -10.2= -0.2
    #    => 残高= 8989.9 + [-0.2 + (10000*0.1)]
    #       = 8989.9 + [-0.2 +1000]
    #       = 8989.9 +999.8= 9989.7
    #
    # => final_balance['cash']= 9989.7
    #
    # history もチェックしておく
    
    # 実際の最終残高との差を比較
    expected_final_balance = 9989.7

    assert abs(final_balance['cash'] - expected_final_balance) < 1e-6, \
        f"Partial close test failed. Got {final_balance['cash']}, expected {expected_final_balance}"

    print("=== PARTIAL CLOSE HISTORY ===")
    for h in history:
        print(h)

    print("=== PARTIAL CLOSE PROFIT HISTORY ===")
    for ph in profit_history:
        print(ph)

    print(f"Test passed: partial close final balance {final_balance['cash']:.2f} matches expected {expected_final_balance:.2f}")

if __name__ == "__main__":
    test_simulation_engine_01()