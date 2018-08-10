import threading
import time
#定义线程需要做的内容，写在函数里面
def target():
    print('当前小弟的线程%s 在运行' % threading.current_thread().name)
    time.sleep(1)
    print('当前小弟的线程 %s 结束' % threading.current_thread().name)

print('当前大哥的线程 %s 在运行' % threading.current_thread().name)
t = threading.Thread(target=target,args = [])

t.start()  #线程启动
#t.join()
print('当前大哥的线程 %s 结束' % threading.current_thread().name)