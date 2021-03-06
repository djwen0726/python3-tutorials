#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:51:18 2018

@author: lisir
"""
import threading
from time import sleep
from time import ctime

loops = [4, 2, 4, 6, 7, 8, 9, 1]


def loop(nloop, nsec):
    print('start loop :', nloop, ' at: ', ctime())
    sleep(nsec)
    print('loop ', nloop, ' done at : ', ctime())


def main():
    print('starting at: ', ctime())
    threads = []
    nloops = range(len(loops))
    for i in nloops:
        t = threading.Thread(target=loop, args=(i,loops[i]))
        threads.append(t)
    # start threads
    for i in nloops:
        threads[i].start()
    # wait for all threads to finish
    for i in nloops:
        threads[i].join()

    print('all done at :', ctime())


if __name__ == '__main__':
    main()
