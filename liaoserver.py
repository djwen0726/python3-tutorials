#!/usr/bin/env python
# -*- coding: utf-8 -*-

'a server example which send hello to client.'

import time, socket, threading

def tcplink(sock, addr):
    print( 'Accept new connection from %s:%s...' % addr)
    sock.send(b'Welcome!')
    while True:
        data = sock.recv(1024)
        time.sleep(1)
        if data == 'exit' or not data:            
            break
        data = 'Hello, %s!!!!!' % data.decode()
        sock.send(data.encode())
    sock.close()
    print( 'Connection from %s:%s closed.' % addr)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 监听端口:
s.bind(('0.0.0.0', 9999))
s.listen(5)
print( 'Waiting for connection...')
sock, addr = s.accept()
print(addr)
print( 'Accept new connection from %s:%s...' % addr)
sock.send(b'Welcome!')

while True:
    data = sock.recv(1024)
    time.sleep(1)
    data=data.decode()
    print(data)
    if data == 'exit' or not data:
        #print( 'Connection from %s:%s closed.' % addr)
        break
    data1 = 'Hello, %s!!!!!' % data
    
    sock.send(data1.encode())

sock.close()
s.close()
print( 'Connection from %s:%s closed.' % addr)
'''
    t = threading.Thread(target=tcplink, args=(sock, addr))
    t.start()
    '''
