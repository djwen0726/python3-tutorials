import threading
import socket

server = socket.socket()
server.bind(("", 8888))
server.listen(10)


def recv_data(conn, addr):
    while True:
        data = conn.recv(1024)
        print(threading.currentThread().name)
        if data:
            print("收到来自 {} 的信息内容 {}".format(addr, data.decode()))
            conn.send(data)
        else:
            print("用户断开了{}的链接".format(addr))
            conn.close()
            break


while True:
    conn, addr = server.accept()

    thread = threading.Thread(target=recv_data, args=(conn, addr))
    thread.start()
