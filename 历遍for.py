li = [1,2,3,4,5,6,7,8,9,10]

for i in li:
     print(li)
     print(i)
     print('队列0是',li[0])
     print('即将删除',li[1])
     li.pop(1)
     print(i)

print(li)
