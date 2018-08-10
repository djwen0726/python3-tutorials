import socket
server = socket.socket()
server.setblocking(False)
server.bind(("", 8888))
server.listen(5)
conn_list = []
while True:
     try:
          conn= server.accept()
          print("用户 {} 已经连接进来了。".format(conn[1]))
          conn[0].setblocking(False)
          conn_list.append(conn)
     except Exception as e:
          pass

     for conn in conn_list:
          try:
              data = conn[0].recv(1024)
              if data:
                   print("接受的数据：{},  来自于{} ".format(data.decode(),conn[1]))
                   print("一共有 {} 个连接,来自第 {} 个连接 ".format(len(conn_list),conn_list.index((conn))))
                   conn[0].sendall(data)
              else:
                    print("用户断开{} 连接，我是第 {} 个连接 ".format(conn[1],conn_list.index((conn))))
                    conn[0].close()
                    conn_list.remove(conn)
          except:
               pass
               

  
