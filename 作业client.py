import socket

client = socket.socket()
client.connect(("127.0.0.1", 8080))
client.setblocking(False)

while True:
    data = input("请输入需要连接的数据内容：")
    if len(data) == 0:
        client.close()
        print("已经关闭客户 {}端连接了！！！".format(client))
        break
    client.send(data.encode())
    print(client.recv(1024).decode())
    if data == 'exit':
        client.close()
        print("已经关闭客户 {}端连接了！！！".format(client))
        break
