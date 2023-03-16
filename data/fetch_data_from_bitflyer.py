import asyncio
from datetime import datetime
import json
import aiohttp
import csv


class chart_1m:
    def __init__(self):
        loop = asyncio.get_event_loop()
        tasks = asyncio.wait([self.get_data()])
        loop.run_until_complete(tasks)
        self.csv_path = 'raw.csv'

    async def get_data(self):
        uri = 'wss://ws.lightstream.bitflyer.com/json-rpc'
        query = {'method': 'subscribe', 'params': {'channel': 'lightning_ticker_FX_BTC_JPY'}}
        while True:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(uri, receive_timeout=60) as client:
                    await asyncio.wait([client.send_str(json.dumps(query))])
                    pre_tick_minute = datetime.utcnow().replace(second=0, microsecond=0)
                    now_minute_price_list = []
                    async for response in client:
                        if response.type != aiohttp.WSMsgType.TEXT:
                            print('response: ' + str(response))
                            break
                        data = json.loads(response[1])['params']['message']
                        data = {'timestamp': datetime.strptime(data['timestamp'][:19], '%Y-%m-%dT%H:%M:%S'), 'ltp': data['ltp']}
                        now_minute = data['timestamp'].replace(second=0, microsecond=0)
                        if pre_tick_minute == now_minute:
                            now_minute_price_list.append(data[('ltp')])
                        elif len(now_minute_price_list) > 0:
                            ohlc = self.make_ohlc(now_minute_price_list)
                            now_minute_price_list = []
                            print(pre_tick_minute, ohlc)
                            with open('raw.csv', 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([pre_tick_minute] + list(ohlc))

                            # print('next minute')
                        # print(data)
                        pre_tick_minute = now_minute


    def make_ohlc(self, price_list):
        open = price_list[0]
        high = max(price_list)
        low = min(price_list)
        close = price_list[-1]
        return open, high, close, low


if __name__ == '__main__':
    chart_1m()