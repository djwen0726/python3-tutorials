import time
import Thread
def countdown(n):
     while n>0:
          n -= 1

count = 50000000

t1=Thread(target=countdown,args=((count//2,)))
t2=Thread(target=countdown,args=((count//2,)))
start=time.time()
##countdown(count)
end=time.time()
print(end-start)

