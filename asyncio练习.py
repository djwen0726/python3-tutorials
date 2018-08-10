import asyncio

async def find_divisibles(inrange, div_by):
    print("finding num in range{} divisible by {}".format(inrange, div_by))
    located = []
    for i in range(inrange):
        if i % div_by == 0:
            located.append(i)
        if i % 500000 == 0:
            await asyncio.sleep(0.0001)

    print("Done",len(located))
    return located

async def main():
    divs1 = loop.create_task(find_divisibles(500000, 301))
    divs2 = loop.create_task(find_divisibles(40000, 389))
    divs3 = loop.create_task(find_divisibles(100, 17))
    await asyncio.wait([divs1, divs2, divs3])

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except Exception as e:
        pass
    finally:
        loop.close()
