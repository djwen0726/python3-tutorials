import queue

q = queue.Queue(6)
print(q.maxsize)
q.put(343)
q.put(23)
q.put(432)

print(q.get())
q.put(44)
q.put(35)
q.put(235)
print(q.full())  #判断队列当前大小是否等于约定队列大小
print(q.qsize())