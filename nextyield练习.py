def foo():
    x = 0
    while True:
        x = yield "yield 数据来了"+str(x)
        print("接受到主线程传来的value:",x)


g = foo()


print("接受到def传来的数据",g.send(None))

print("接受到def传来的数据",g.send(100))

print("接受到def传来的数据",g.send(200))

