import socket
import selectors

server = socket.socket()
server.bind(("", 8885))

server.listen(10)

epoll_selector = selectors.EpollSelector()


def recv_data(conn):
    data = conn.recv(1024)
    if data:
        print(data.decode())
        conn.send(data)
    else:
        conn.close()
        epoll_selector.unregister(conn)
        print("客户端连接 关闭")


def connection(server):
    conn, addr = server.accept()
    print("用户{}连接".format(addr))
    epoll_selector.register(conn, selectors.EVENT_READ, recv_data)


epoll_selector.register(server, selectors.EVENT_READ, connection)

while True:

    events = epoll_selector.select()

    for key, mask in events:
        print(type(key))
        print(key.fileobj, key.fd, key.events, key.data)
        func = key.data
        sock = key.fileobj
        func(sock)
