import threading as th
import random
import queue
import time


class Producer(th.Thread):
    def __init__(self, queue):
        th.Thread.__init__(self)
        self.queue = queue

    def run(self):
        for i in range(10):
            data = random.randint(0, 100)
            print("生产者生产了：{} ,放入队列中".format(data))
            self.queue.put(data)
            time.sleep(1)
        print("生产者完成了所有生产。")


class Consumer(th.Thread):
    def __init__(self, queue):
        th.Thread.__init__(self)
        self.queue = queue

    def run(self):
        for i in range(10):
            item = self.queue.get()
            print("消费者从队列拿到：", item)
        print("消费者完成了所有的消费。")

# class Consumer(th.Thread):
#     def __init__(self,  queue):
#         th.Thread.__init__(self)
#         self.data = queue
#
#     def run(self):
#         for i in range(10):
#             val = self.data.get()
#             print("消费者正在消费. %d 在队列中被消费!" % val)
#             # time.sleep(random.randrange(10))
#         print("%s finished!" % self.getName())

#主程序
# def main():
#     # global queue
#     queue = queue.Queue(5)
#     p = Producer(queue)
#     c = Consumer(queue)
#     p.start()
#     c.start()
#     p.join()
#     c.join()
#
#     print('所有的线程都已经完成了!')


if __name__ == '__main__':

    print(queue)
    # main()
    print(time)
    print(random)
