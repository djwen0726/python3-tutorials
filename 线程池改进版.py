import time
import threading
import queue


def func1(*args, **kwargs):
    time.sleep(3)
    print("任务一完成", args, kwargs)


def func2(*args, **kwargs):
    time.sleep(1)
    print("任务二完成", args, kwargs)


class ThreadPool():
    def __init__(self, n):

        self.queue = queue.Queue()
        for i in range(n):
            print(i)
            threading.Thread(target=self.run, daemon=True).start()

    def worker(self):
        while True:
            task, args, kwargs = self.queue.get()
            task(*args, **kwargs)
            self.queue.task_done()

    def run(self):
        while True:
            print("我已经在run里面了",threading.currentThread())
            task, args, kwargs = self.queue.get()
            print("从队列中获得数据", task, "拿到的参数是：", args, kwargs,threading.currentThread())
            # print("拿到的参数是：", args, kwargs)
            task(*args, **kwargs)
            self.queue.task_done()

    def apply_async(self, func, *args, **kwargs):
        self.queue.put((func, args, kwargs))

    def join(self, timeout=None):
        self.queue.join()


thread = ThreadPool(5)
time.sleep(10)
thread.apply_async(func1, 1)
time.sleep(10)
thread.apply_async(func1, 2)
thread.apply_async(func1, 3)
thread.apply_async(func1, 4)
thread.apply_async(func1, 5)
thread.apply_async(func1, 6)
thread.join()
'''
1. 阻塞主进程，专注于执行多线程中的程序。

2. 多线程多join的情况下，依次执行各线程的join方法，前头一个结束了才能执行后面一个。

3. 无参数，则等待到该线程结束，才开始执行下一个线程的join。

4. 参数timeout为线程的阻塞时间，如 timeout=2 就是罩着这个线程2s 以后，就不管他了，继续执行下面的代码。
'''
