from socket import *
from fib import fib
from threading import Thread


def fib_server(address):
    sock = socket(family=AF_INET, type=SOCK_STREAM)
    sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

    sock.bind(address)
    sock.listen(5)

    while True:
        client, addr = sock.accept()
        print("connection", addr)
        Thread(target=fib_handler, args=(client,))


def fib_handler(client):
    while True:
        req = client.recv(100)
        if not req:
            break
        n = int(req)
        result = fib(n)
        resp = str(result).encode(encoding='utf-8') + b'\n'
        client.send(resp)
    print("closed")


fib_server(('', 25000))
