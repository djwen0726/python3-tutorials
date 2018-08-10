import socket
mysock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
mysock.connect(("127.0.0.1",5555))
mysock.sendto(b"hello from client",("127.0.0.1",5555))
