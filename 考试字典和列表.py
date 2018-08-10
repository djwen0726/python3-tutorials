
dic1={"k1":"v1", "k2":"v2","k3":"v3", "k4":"v4"}

list1=["l1", "l2", "l3"]


def fun(li,dic):

     li1 = tuple(dic.values())

##     dic1 = dict(zip(tuple((dic.keys())),li))

     dic1 = dict(zip(dic,li))

     return li1,dic1


(li,dic) = fun(list1,dic1)

print('li=',li)

print('dic=',dic)
