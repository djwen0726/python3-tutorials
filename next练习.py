import time

def generator():
    while True:
        print("进入def")
        receive = yield 1
        time.sleep(1)
        print("def")
        print('def 里面的extra' + str(receive))


g = generator()
print("主程序next（g）",next(g))
time.sleep(3)
print("主程序准备send")
print("主程序g.send(111)",g.send(111))
print("-------------")
print("主程序第二次next（g）",g.send(222))
print("===========================")
