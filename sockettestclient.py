import socket
mysocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
mysocket.connect(("47.75.45.141",8881))
mysocket.send('机器猫'.encode())

