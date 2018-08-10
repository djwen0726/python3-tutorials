#!/usr/bin/env python
# -*- coding: utf-8 -*-

'a socket example which send echo message to server.'

import socket,time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 建立连接:
s.connect(('127.0.0.1', 8889))
# 接收欢迎消息:
#print( s.recv(1024).decode())
for data in ['Michael', 'Tracy', 'Sarah','机器猫']:
    # 发送数据:
    s.send(data.encode())
    print( s.recv(1024).decode())



s.send(b'exit')


s.close()

