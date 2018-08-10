def consumer():
    r = '初始状态'
    n = 100
    while True:
        n = yield r
        if not n:
            return
        print('[消费者] 正在消费 %s...' % n)
        r = '200 OK'+str(n)

def produce(c):
    print("第一次send后返回的结果值",c.send(None))
    n = 0
    while n < 5:
        n = n + 1
        print('[生产者] 正在生产 %s...' % n)
        r = c.send(None)
        # next(c)
        print('[生产者] 消费者返回: %s' % r)
    c.close()

c = consumer()
produce(c)