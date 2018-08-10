def my_gen():
    print("第一次执行")
    a= yield 1
    print("-----------------------")
    print(a)
    print("第二次执行")
    a=yield 2
    print(a)
    print("第三次执行")
g=my_gen()
v1=g.send(None)
print(v1)
v2=g.send(None)
print(v2)
v3=next(g)
print((v3))

