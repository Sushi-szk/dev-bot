import asyncio
import aiohttp
import csv
import os
from datetime import datetime, timedelta

async def fetch_kline_chunk(session, base_url, params, semaphore, delay):
    """1つのデータ範囲を取得"""
    async with semaphore:  # 同時リクエスト数を制限
        await asyncio.sleep(delay)  # 各リクエストの間隔を調整
        async with session.get(base_url, params=params) as response:
            if response.status != 200:
                print(f"Failed to fetch data: {response.status}, {await response.text()}")
                return []
            json_data = await response.json()
            return json_data.get("result", {}).get("list", [])

async def fetch_kline_data(symbol, interval, start_date, end_date, file_name):
    base_url = "https://api.bybit.com/v5/market/kline"
    data = []
    tasks = []

    # ディレクトリを作成（存在しない場合のみ）
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # セッションを一度だけ作成
    connector = aiohttp.TCPConnector(limit=50)  # コネクション制限を調整
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(20)  # 同時リクエスト数を20に制限
        current_start = start_date
        delay = 0.5  # リクエスト間隔を0.5秒に設定

        while current_start < end_date:
            current_end = current_start + timedelta(minutes=interval * 200)
            if current_end > end_date:
                current_end = end_date

            # UNIXタイムスタンプ（ミリ秒）に変換
            start_timestamp = int(current_start.timestamp() * 1000)
            end_timestamp = int(current_end.timestamp() * 1000)

            print(f"Preparing task for {current_start} to {current_end}")

            # リクエストパラメータ
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": str(interval),
                "start": start_timestamp,
                "end": end_timestamp,
                "limit": 200
            }

            # タスクをリストに追加
            tasks.append(fetch_kline_chunk(session, base_url, params, semaphore, delay))

            # 次の範囲に進む
            current_start = current_end

        # すべてのタスクを並行して実行
        for task in asyncio.as_completed(tasks):
            result = await task
            if isinstance(result, list):
                data.extend(result)
            else:
                print(f"Error encountered: {result}")

    # 重複を削除し、データを古い順に並べ替え
    unique_data = list({tuple(item): item for item in data}.values())
    unique_data.sort(key=lambda x: int(x[0]))  # タイムスタンプでソート

    # 日付を追加
    formatted_data = []
    for row in unique_data:
        timestamp = int(row[0])
        readable_date = datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')  # UTCで変換
        formatted_data.append([timestamp, readable_date] + row[1:])

    # CSVに保存
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "date", "open", "high", "low", "close", "volume"])
        writer.writerows(formatted_data)

    print(f"Data saved to {file_name}")

if __name__ == "__main__":
    symbol = "BTCUSDT"  # 通貨ペア
    interval = 5  # 分足
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 1)
    file_name = f"kline/{symbol}_{interval}m.csv"

    asyncio.run(fetch_kline_data(symbol, interval, start_date, end_date, file_name))