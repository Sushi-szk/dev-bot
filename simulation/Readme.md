# SimulationEngine
本ドキュメントでは、本リポジトリに含まれる SimulationEngine クラスの仕組み・目的・利用方法などを詳細に解説します。
このエンジンは、過去の価格データを用いたバックテストを行うために設計されており、複数のポジション管理や部分クローズ、指値注文、テイクプロフィット/ストップロスなどの機能をサポートします。特徴は以下になります。
- 過去データ(price_data) と Strategy を用意するだけで、複数アクションのシミュレーション を実行可能
- 部分クローズ などの細かいロジックも再現
- 戦略ロジック(キャンセル/アクション) → エンジン(約定/ポジション管理) → 結果保存 の責務分離により、柔軟に戦略を差し替えられる
- 実運用システムに近い形でバックテストを回したい場合や、細かい発注ロジック、証拠金管理を再現したい場合は、現行の仕組みをさらに拡張することで柔軟に対応

## 概要
SimulationEngine は、以下の主な機能を提供するバックテスト用シミュレーションエンジンです。

- 成り行き注文 / 指値注文
- 指値注文の場合、価格が到達した時点でポジションをエントリー(買い/売り)
- テイクプロフィット / ストップロス (TP/SL) 自動判定
- ポジションに設定されている TP/SL の価格に相場が達した場合、自動でポジションをクローズし利益確定 or 損切り
- 複数アクション (buy, sell, close_long, close_short...) を同一ステップで実行
- Strategy 側が複数のアクションを返すと、同じローソク足(ステップ)内で順次実行
- 部分クローズ対応
- クローズ時にクローズサイズを指定でき、ポジションの一部だけを決済することが可能
- 手数料計算
- エントリー時およびクローズ時に手数料を計算し、実現損益(net_pnl)を正しく反映
- 累積利益やトレード履歴の保存
- 各ステップで生じた実現損益を累積し、最終的に profit_history に履歴を持つ
- 取引の発生履歴(エントリー/クローズなど)を history に保存し、CSV 出力できる
- バックテスト完了後に最終的な損益やドローダウン、シャープレシオなどの指標を自動で算出します。

## ファイル・クラス構造
### SimulationEngine

- バックテストの本体。
- コンストラクタ(__init__) で初期残高・価格データ・手数料関数などを受け取り、シミュレーションの準備をする
- run_simulation(strategy, lookback_period) を呼ぶと、ローソク足を1本ずつ進めながらトレードを実行
- 全部完了後に generate_metrics() で最終的な損益・ドローダウン・シャープレシオを表示
- save_results() で CSV (トレード履歴 / 損益履歴) を出力

### fixed_fee(amount) (サンプル)
- 取引額に対して 1% の固定手数料を返す例

### MultiActionStrategy (サンプル)
- 複数アクション (buy / sell / close_long / close_short ...) を同じステップで返すデモ用の戦略クラス
### test_simulation()
- 簡易テスト用メソッド。簡単な価格データを生成し、エンジンを動かすデモとして利用

## SimulationEngine主要なメソッドの解説
1. コンストラクタ: __init__
    ```python
    def __init__(
        self, 
        initial_balance: float, 
        price_data: pd.DataFrame, 
        fee_function: Callable[[float], float], 
        output_dir: str = 'runs',
        market_data_function: Optional[Callable[[int, int], pd.DataFrame]] = None
    ) -> None:
    ```
    - initial_balance: シミュレーションの初期キャッシュ残高
    - price_data: 過去の価格データ (pandas DataFrame)
    - 少なくとも timestamp, open, high, low, close 列が必要
    - fee_function: 取引手数料を計算する関数。amount -> fee のように実装
    - output_dir: トレード履歴や損益履歴を出力するディレクトリ(デフォルトは "runs")
    - market_data_function: 過去データをどう切り出すかを定義する関数。デフォルトでは default_market_data_function を使用

    この段階で、下記のようなアトリビュートを初期化します:
    - self.balance: {'cash': initial_balance} で、現在のキャッシュ残高を管理
    - self.positions: ポジションをリスト形式で保持 (後述)
    - self.open_orders: 未約定の指値注文リスト (後述)
    - self.history: 取引履歴(ログ)
    - self.profit_history: ステップごとに「累計実現損益」を記録
    - self.total_fees: 累積手数料

2. run_simulation(strategy, lookback_period)
シミュレーションのメインループ。価格データ(price_data)の各行(ローソク足)を1ステップとして反復し、下記の順序で処理します  
    1. process_open_orders(row)  
        指値注文が約定可能かを判定( row['low'], row['high'] )
        約定する場合は _execute_market_order(...) を呼び出し、ポジションを持つ
    1. check_positions_tp_sl(row)  
        すでに保有しているポジションの TP/SL 判定
        該当すれば close_position(...) を呼んで実現損益を加算
    1. strategy.cancel_orders(...)  
        Strategy が「特定の注文をキャンセルしたい」と返すIDを受け取り、cancel_order(...) で取り消し
    1. strategy(...) によるアクション実行  
        戻り値が複数の (action, size, limit_price, tp, sl) のリストであれば順次処理
        例: ('buy', 0.1, None, None, None) → submit_order('buy', ...)
        ステップごとの実現損益を累計し、profit_history に記録

    シミュレーションが終了したら generate_metrics() を呼び出して最終結果を表示し、save_results() で CSV 出力
    戻り値として balance, history, profit_history を返す。

3. 注文関連メソッド
    1. submit_order(...)  
    `(side, size, limit_price, take_profit, stop_loss)`  
    limit_price があれば指値注文として open_orders に登録
    なければ成り行き注文として _execute_market_order(...) を即時実行
    1. _execute_market_order(...)    
    'buy' or 'sell' を判定し、execute_buy(...)/execute_sell(...) を呼び出す
    3. cancel_order(order_id)  
    open_orders から該当IDの注文を削除
    4. process_open_orders(row)  
    open_orders を走査し、limit_price が row['low'] ~ row['high'] の範囲に達している注文を約定(= _execute_market_order)
    約定した注文は open_orders から取り除く

4. ポジション管理
    1. execute_buy(...) / execute_sell(...)  
    新規 or 既存ポジションのエントリー  
    成り行き(または指値約定時)に呼び出される  
    ロングの場合(execute_buy):  
    cost = price * size 分 + 手数料 を balance['cash'] から引く   
    既存ロングがあれば平均取得単価を再計算して追加、なければ新規作成  
    entry_fee に手数料を加算  
    ショートの場合(execute_sell):  
    同様に cost + fee 分を証拠金として口座残高から引く(シンプル化モデル)  
    ショートポジションを追加 or 既存ショートにサイズ上乗せ  
    2. close_position(pos_idx, close_price, close_size=...)  
    部分クローズ対応  
    close_size が pos_size より小さければ一部のみ決済し、残りを保持
    close_size が pos_size に等しければ全クローズ  
    損益計算:  
    gross_pnl = (close_price - entry_price) * close_size (ロングの場合; ショートは逆)  
    close_fee = fee_function(...)  
    partial_entry_fee = entry_fee * (close_size / pos_size) で按分  
    net_pnl = gross_pnl - partial_entry_fee - close_fee  
    口座残高 には、net_pnl + partial_entry_fee + (entry_price * close_size)(entry_feeを一部戻したうえで差し引くという実装ポリシー; シンプル化モデル)  
    取引履歴(history) にクローズ情報を追加  
5. TP/SL(テイクプロフィット/ストップロス)判定
    ```python
    def check_positions_tp_sl(self, current_row: pd.Series) -> float:
        # ...
    ```
    ローソク足の high, low とポジションの take_profit / stop_loss を比較  
    ロング なら  
    sl is not None and low_ <= sl → 損切り  
    tp is not None and high_ >= tp → 利確  
    同時到達なら損切り優先  
    ショートは逆  
    引っかかったポジションは close_position(...) して損益を計上  
    同メソッドを各ステップで呼び出し、自動的にポジションをクローズ  
6. 損益履歴・メトリクス
    1. profit_history  
    「実現損益の累計」を各ステップで {'timestamp':..., 'profit':...} として記録
    バックテストが進むごとに step_pnl を加算
    2. generate_metrics()  
    最終的に profit_history から
    最終損益 (final_net_profit)
    最大ドローダウン (max_drawdown)
    シャープレシオ (sharpe_ratio)を計算して表示する
    シャープレシオ計算は簡易的にprofit_series.pct_change() から算出、表示後、結果は標準出力に出力される
7. 保存処理: save_results()  
    history を trade_history.csv に出力
    profit_history を profit_history.csv に出力
    ディレクトリは output_dir 引数(デフォルトは 'runs')

## 実行の流れ
エンジン初期化:
```python
engine = SimulationEngine(
    initial_balance=10000.0,
    price_data=some_price_data,
    fee_function=fixed_fee
)
```
ストラテジー(戦略)定義:
```python
class MyStrategy:
    def cancel_orders(self, open_orders, market_data, balance, positions):
        return []
    def __call__(self, market_data, balance, positions):
        return [('buy', 0.1, None, None, None)]  # 例: 毎ステップ0.1BTC buy
```
シミュレーション実行:
```python
final_balance, history, profit_history = engine.run_simulation(
    strategy=MyStrategy(),
    lookback_period=10
)
```
結果の確認:  
コンソール上に最終損益やシャープレシオ等が表示され、output_dir に CSV ファイルが保存  
