import multiprocessing as mp
import threading as th
import socket

#process = mp.Process(target=func, conn=cxx)

server = socket.socket()
server.bind(("", 8888))
server.listen(5)


class MyProcess(mp.Process):
    def __init__(self, conn, addr):
        super().__init__()
        self.conn = conn
        self.addr = addr

    def run(self):
        while True:
            data = self.conn.recv(1024)
            if data:
                print("接受到来自{} 的数据信息{}".format(self.addr, data.decode()))
                self.conn.send(data)
            else:
                print("断开到来自{} 的连接".format(self.addr))
                self.conn.close()
                break


while True:
    conn, addr = server.accept()
    process = MyProcess(conn, addr)
    process.start()
