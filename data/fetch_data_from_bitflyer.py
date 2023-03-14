import asyncio
import json
import aiohttp


class chart_1m:
    data_list = []

    def __init__(self):
        loop = asyncio.get_event_loop()
        tasks = asyncio.wait([self.get_data(), self.test()])
        loop.run_until_complete(tasks)

    async def get_data(self):
        uri = 'wss://ws.lightstream.bitflyer.com/json-rpc'
        query = {'method': 'subscribe', 'params': {'channel': 'lightning_ticker_FX_BTC_JPY'}}
        while True:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(uri, receive_timeout=60) as client:
                    await asyncio.wait([client.send_str(json.dumps(query))])
                    async for response in client:
                        if response.type != aiohttp.WSMsgType.TEXT:
                            print('response: ' + str(response))
                            break
                        data = json.loads(response[1])['params']['message']
                        # print(data)
                        self.data_list.append(data)
                    print('a')

    async def test(self):
        while True:
            if self.data_list:
                print(self.data_list)
                self.data_list = []
                await asyncio.sleep(3)
