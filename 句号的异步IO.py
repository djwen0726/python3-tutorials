import asyncio
import time
from datetime import datetime

async def test(x):
    print("x is :",x)
    await asyncio.sleep(x)
    print("第二次x is :",x)
    return "返回了"+str(x)

print("start",datetime.now())

tasks = [asyncio.ensure_future(test(i)) for i in range(5,0,-1)]
loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.wait(tasks))

for task in tasks:
    print(task.result())

print("end:",datetime.now())