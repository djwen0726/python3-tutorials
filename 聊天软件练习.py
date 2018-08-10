import socket
import threading


def recv_data():
    while True:
        data = client.recv(1024)
        print(data.decode())



client = socket.socket()
client.connect(('127.0.0.1', 8080))

thread = threading.Thread(target=recv_data, daemon=True)
thread.start()

while True:
    a = input('请输入聊天内容：')
    client.send(a.encode())