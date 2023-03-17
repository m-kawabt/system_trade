import asyncio
import csv
import json
from datetime import datetime as dt
from datetime import timedelta as td


import aiohttp
import pandas as pd
import mplfinance as mpf


class chart_1m:
    def __init__(self):
        self.realtime_data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close'])
        self.realtime_data.index.name = 'Time'
        loop = asyncio.get_event_loop()
        tasks = asyncio.wait([self.get_data(), self.make_chart()])
        loop.run_until_complete(tasks)

    async def get_data(self):
        uri = 'wss://ws.lightstream.bitflyer.com/json-rpc'
        query = {'method': 'subscribe', 'params': {'channel': 'lightning_ticker_FX_BTC_JPY'}}
        while True:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(uri, receive_timeout=60) as client:
                    await asyncio.wait([client.send_str(json.dumps(query))])
                    pre_tick_minute = dt.now().replace(second=0, microsecond=0)
                    now_minute_price_list = []
                    async for response in client:
                        if response.type != aiohttp.WSMsgType.TEXT:
                            print('response: ' + str(response))
                            break
                        data = json.loads(response[1])['params']['message']
                        data = {'timestamp': dt.strptime(data['timestamp'][:19], '%Y-%m-%dT%H:%M:%S'), 'ltp': data['ltp']}
                        data['timestamp'] = data['timestamp'] + td(hours=9)
                        now_minute = data['timestamp'].replace(second=0, microsecond=0)
                        if pre_tick_minute != now_minute and len(now_minute_price_list) > 0:
                            ohlc = self.make_ohlc(now_minute_price_list)
                            now_minute_price_list = []
                            self.realtime_data.loc[pre_tick_minute] = list(ohlc) # type: ignore
                            with open('raw.csv', 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([pre_tick_minute] + list(ohlc))
                        now_minute_price_list.append(data[('ltp')])
                        pre_tick_minute = now_minute

    async def make_chart(self):
        while True:
            if not self.realtime_data.empty:
                mpf.plot(self.realtime_data, type='candle', savefig='data/realtime_chart.png')
            await asyncio.sleep(5)


    def make_ohlc(self, price_list):
        open = price_list[0]
        high = max(price_list)
        low = min(price_list)
        close = price_list[-1]
        return open, high, low, close


if __name__ == '__main__':
    chart_1m()