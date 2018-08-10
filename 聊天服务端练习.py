from multiprocessing import Pool, cpu_count,Manager
from multiprocessing.pool import ThreadPool
import socket


#从队列中拿出数据，发给所有连接上的客户端
def send_data(dict_proxy, queue_proxy):
    while True:
        data = queue_proxy.get()        
        print(data.decode())
        for conn in dict_proxy.values():
            conn.send(data)



def worker_thread(connection, addr, dict_proxy, queue_proxy):
    while True:
        try:
            recv_data = connection.recv(1024)
            if recv_data:
                data = "来自{addr} 的消息:{data}".format(addr=addr, data=recv_data.decode())
                queue_proxy.put(data.encode())     #把消息添加到到队列中
            else:
                raise Exception
        except:
            dict_proxy.pop(addr)                #从字典中删掉退出的客户端
            data = '用户{}退出'.format(addr)
            queue_proxy.put(data.encode())      #把退出消息添加到队列中
            connection.close()
            break



def worker_process(server, dict_proxy, queue_proxy):
    thread_pool = ThreadPool( cpu_count()*2 )   #通常分配2倍CPU个数的线程
    thread_pool.apply_async(send_data, args=(dict_proxy, queue_proxy))
    while True:
        conncetion, remote_address = server.accept()        
        dict_proxy.setdefault(remote_address, conncetion)   #把套接字加入字典中

        data = '用户{}登录'.format(remote_address)
        queue_proxy.put(data.encode())                      #将用户登录消息添加到队列中
        thread_pool.apply_async(worker_thread, args=(conncetion, remote_address, dict_proxy, queue_proxy))


if __name__ == '__main__':

    server = socket.socket()
    server.bind(('', 8080))
    server.listen(1000)

    mgr = Manager()
    dict_proxy = mgr.dict()         #用来保存连接上来的客户端，
    queue_proxy = mgr.Queue()       #把客户端发过来的消息通过队列传递

    n = cpu_count()                 #打印当前电脑的cpu核数
    process_pool = Pool(n)
    for i in range(n):              #充分利用CPU，为每一个CPU分配一个进程
        process_pool.apply_async(worker_process, args=(server, dict_proxy, queue_proxy))    #把server丢到两个进程里面

    process_pool.close()
    process_pool.join()