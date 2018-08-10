# /usr/bin/env python
# coding:utf-8

__author__ = 'kikay'

import threading

# Threading.local对象
ThreadLocalHelper = threading.local()
lock = threading.RLock()


class MyTheadEx(threading.Thread):
    def __init__(self, threadName, name, age, sex):
        super(MyTheadEx, self).__init__(name=threadName)
        self.__name = name
        self.__age = age
        self.__sex = sex

    def run(self):
        global ThreadLocalHelper
        ThreadLocalHelper.ThreadName = self.name
        ThreadLocalHelper.Name = self.__name
        ThreadLocalHelper.Age = self.__age
        ThreadLocalHelper.Sex = self.__sex
        MyTheadEx.ThreadPoc()

    # 线程处理函数
    @staticmethod
    def ThreadPoc():
        lock.acquire()
        try:
            print('Thread={id}'.format(id=ThreadLocalHelper.ThreadName))
            print('Name={name}'.format(name=ThreadLocalHelper.Name))
            print('Age={age}'.format(age=ThreadLocalHelper.Age))
            print('Sex={sex}'.format(sex=ThreadLocalHelper.Sex))
            print('----------')
        finally:
            lock.release()


if __name__ == '__main__':
    Tom = {'Name': 'tom', 'Age': 20, 'Sex': 'man'}
    xiaohua = {'Name': 'xiaohua', 'Age': 18, 'Sex': 'woman'}
    Andy = {'Name': 'Andy', 'Age': 40, 'Sex': 'man'}
    T = (Tom, xiaohua, Andy)
    threads = []
    for i in range(len(T)):
        t = MyTheadEx(threadName='id_{0}'.format(i), name=T[i]['Name'], age=T[i]['Age'], sex=T[i]['Sex'])
        threads.append(t)
    for i in range(len(threads)):
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()
    print('All Done!!!')
