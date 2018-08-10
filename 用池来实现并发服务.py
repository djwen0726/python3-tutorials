from multiprocessing.pool import ThreadPool
from multiprocessing import Pool,cpu_count
import socket

server = socket.socket()
server.bind(("", 8888))
server.listen(100)

def func(conn, addr):
    while True:
        recv_data = conn.recv(1024)
        if recv_data:
            print("接受到来自{}的消息{}".format(addr, recv_data.decode()))
            conn.send(recv_data)
        else:
            conn.close()
            print("用户已经中断连接")
            break

def accept(server):
    pool = ThreadPool(cpu_count()*2)
    while True:
        conn,addr = server.accept()
        pool.apply_async(func,args=(conn,))

n =cpu_count()

pool = ThreadPool(n)

while True:
    conn, addr = server.accept()
    pool.apply_async(func, args=(conn, addr))
