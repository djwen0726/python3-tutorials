import queue
import random, threading, time
# 生产者类
class Producer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.data = queue

    def run(self):
        for i in range(10):
            data=random.randint(0,100)
            print("%s is producing %d to the queue!" % (self.getName(), data))
            self.data.put(data)
            time.sleep(1)
        print("%s finished!" % self.getName())
# 消费者类
class Consumer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.data = queue

    def run(self):
        for i in range(10):
            val = self.data.get()
            print("%s is consuming. %d in the queue is consumed!" % (self.getName(), val))
            # time.sleep(random.randrange(10))
        print("%s finished!" % self.getName())
#主程序
def main():
    global queue
    queue = queue.Queue()
    producer = Producer('制造者', queue)
    consumer = Consumer('消费者', queue)

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()
    print('All threads finished!')


if __name__ == '__main__':
    main()
