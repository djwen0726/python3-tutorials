from multiprocessing.pool import ThreadPool

import time


def fun1(*args, **kwargs):
    time.sleep(3)
    print("任务一完成", args, kwargs)


def func2(*args, **kwargs):
    time.sleep(1)
    print("任务二完成", args, kwargs)


pool = ThreadPool(4)

pool.apply_async(fun1,args=(1,))

pool.apply_async(fun1,args=(2,))
pool.apply_async(fun1,args=(3,))
pool.apply_async(fun1,args=(4,))
pool.apply_async(fun1,args=(5,))
pool.apply_async(fun1,args=(6,))
pool.apply_async(fun1,args=(7,))

pool.close()
pool.join()