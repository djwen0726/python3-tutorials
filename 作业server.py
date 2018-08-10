import socket

server = socket.socket()


server.bind(("",8889))

server.listen(5)

while True:
     conn,addr = server.accept()
     print("用户 {} 连接过来了!!".format(addr))
     while True:
          try:
               data = conn.recv(1024)
               if data:
                    print("收到的数据是:",data.decode())
                    conn.send(data)
               else:
                    print("客户端{}已经断开!!".format(addr))
                    conn.colse()
                    break
          except Exception as e:
               print("意外故障，客户端{}已经断开连接".format(addr))
               conn.close()
               break
