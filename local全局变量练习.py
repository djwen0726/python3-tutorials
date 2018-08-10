import threading

global_num = 0
lock = threading.Lock()


def thread_cal():
    lock.acquire()
    global global_num
    for i in range(1000):
        global_num += 1
    print('当前小弟的线程{} 在运行,结果是{}'.format(threading.current_thread().getName(), global_num))
    lock.release()


# Get 10 threads, run them and wait them all finished.
# lock = threading.Lock()
threads = []
for i in range(10):
    threads.append(threading.Thread(target=thread_cal))
    threads[i].start()
for i in range(10):
    threads[i].join()

# Value of global variable can be confused.
print(global_num)
