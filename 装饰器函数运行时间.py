
import time
 
def deco(func):
    startTime = time.time()
    func()
    endTime = time.time()
    msecs = (endTime - startTime)*1000
    print("time is %d ms" %msecs)
 
 
def func():
    print("hello")
    time.sleep(1)
    print("world")
 
if __name__ == '__main__':
    f = func
    deco(f)#只有把func()或者f()作为参数执行，新加入功能才会生效
    print("f.__name__ is",f.__name__)#f的name就是func()
    print()
