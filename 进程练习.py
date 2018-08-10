import threading

local = threading.local()


def func(name):
    print('current thread:%s' % threading.currentThread().name)
    local.name = name
    print("%s in %s" % (local.name, threading.currentThread().name))


t1 = threading.Thread(target=func, args=('第1号进程',))
t2 = threading.Thread(target=func, args=('第2号进程',))
t1.start()
t2.start()
t1.join()
t2.join()
