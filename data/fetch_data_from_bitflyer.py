import asyncio
import json
import aiohttp
from time import sleep


async def get_data():
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
                    data = json.loads(response[1])
                    print(data)

loop = asyncio.get_event_loop()
loop.run_until_complete(get_data())