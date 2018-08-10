import time
import threading
import queue


def func1(*args, **kwargs):
    time.sleep(1)
    print("任务一完成",args, kwargs)


def func2(*args, **kwargs):
    time.sleep(1)
    print("任务二完成",args, kwargs)


class MyThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.queue = queue.Queue()
        self.start()

    def run(self):
        while True:
            task, args, kwargs = self.queue.get()
            self.queue.task_done()
            print("从队列中获得数据", task,"拿到的参数是：", args, kwargs)
            # print("拿到的参数是：", args, kwargs)
            task(*args, **kwargs)

    def apply_async(self, func, *args, **kwargs):
        self.queue.put((func, args, kwargs))

    def join(self, timeout=None):
        self.queue.join()


thread = MyThread()
thread.apply_async(func1)
thread.apply_async(func2, 1, 2, 3,a=1,b=2,c=4)
thread.apply_async(func1)
thread.join()
thread.join()
thread.join()
# print(help(thread))
'''
1. 阻塞主进程，专注于执行多线程中的程序。

2. 多线程多join的情况下，依次执行各线程的join方法，前头一个结束了才能执行后面一个。

3. 无参数，则等待到该线程结束，才开始执行下一个线程的join。

4. 参数timeout为线程的阻塞时间，如 timeout=2 就是罩着这个线程2s 以后，就不管他了，继续执行下面的代码。
'''
