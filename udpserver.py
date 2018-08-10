import socket
mysock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
mysock.bind(("127.0.0.1",5555))
clientdata,addr = mysock.recvfrom(2048)
print(clientdata.decode())
print(addr)
