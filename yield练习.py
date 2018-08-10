def my_gen():
    print("第一次执行")
    result = yield 1
    print("result", result)
    print("第二次执行")
    result = yield 2
    print(result)
    # print("result", result)
    print("第三次执行")
    result = yield 3
    # print("result", result)


g = my_gen()
print(g)
a=next(g)
#
print(a)




# g.send(999)

for i in g:
    print(i)
    next(g)
    g.send("999")
    print("----------------")
