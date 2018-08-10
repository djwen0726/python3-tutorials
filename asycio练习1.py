import time
import asyncio

async def is_prime(x):
    return  not any(x//i == x/i for i in range(x-1,1,-1))

async def highest_prime_below(x):
    print("最大的素数低于：%d"%x)
    for y in range(x-1,0,-1):
        if await is_prime(y):
            print("最大的素数 %d在 %d"%(x,y))
            return  y
        await asyncio.sleep(0.0001)

    return None

async def main():
    t0 = time.time()
    await asyncio.wait([
        highest_prime_below(900000),
        highest_prime_below(910000),
        highest_prime_below(91000)
    ])
    t1 = time.time()
    print("Took %.2f ms" %(1000*(t1-t0)))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

# if __name__ == "__main__":
#
#     main()