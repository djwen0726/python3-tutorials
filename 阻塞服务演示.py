import socket



server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(("0.0.0.0",8888))
server.listen()
while True:  # 不断接收新连接
    conn,addr = server.accept()  # 阻塞
    while True:  # 接收连接，多次通信
        print("new conn",addr)
        data = conn.recv(1024)  #官方建议最大8192
        conn.send(data.upper())
        # recv 默认是阻塞的
        if not data :  
            break  # 客户端一断开，conn.recv 收的的就都是空数据，就进入死循环了
# 只能同时服务一个连接
server.close()
